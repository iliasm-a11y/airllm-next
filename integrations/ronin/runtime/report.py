"""Source-backed local reports with explicit uncertainty and redacted exports."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from xml.sax.saxutils import escape
import zipfile


SECRET_KEYS = re.compile(r"(?i)(?:password|passwd|secret|authorization|cookie|api[_-]?key|access[_-]?token|refresh[_-]?token|private[_-]?key)")
STATUS_LABELS = {
    "hypothesis": "Hypothèse non vérifiée", "hypothese": "Hypothèse non vérifiée",
    "clue": "Indice", "observation": "Observation", "reproduced": "Reproduit",
    "confirmed": "Confirmé selon le protocole enregistré", "not_reproduced": "Non reproduit",
    "rejected": "Rejeté", "reproduit": "Reproduit", "confirmed_lab": "Confirmé dans le laboratoire local",
}


def redact_text(value):
    text = str(value)
    text = re.sub(r"-----BEGIN [^-]*(?:PRIVATE KEY|CERTIFICATE)-----.*?-----END [^-]+-----", "[BLOC SENSIBLE EXPURGÉ]", text, flags=re.S)
    text = re.sub(r"(?im)^((?:authorization|proxy-authorization|cookie|set-cookie)\s*[:=]\s*).*$", r"\1[EXPURGÉ]", text)
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [EXPURGÉ]", text)
    text = re.sub(r"(?i)([\"']?(?:password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret)[\"']?\s*[:=]\s*)(?:[\"'][^\"'\n]*[\"']|[^\s,;&}\n]+)", r"\1[EXPURGÉ]", text)
    text = re.sub(r"(?i)([?&](?:token|key|secret|password|signature|auth)=)[^&#\s]+", r"\1[EXPURGÉ]", text)
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{15,}|github_pat_[A-Za-z0-9_]{15,}|AKIA[A-Z0-9]{16})\b", "[IDENTIFIANT EXPURGÉ]", text)
    text = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[IDENTIFIANTS EXPURGÉS]@", text)
    text = re.sub(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL EXPURGÉ]", text)
    text = text.replace(str(Path.home()), "~")
    return text


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): "[EXPURGÉ]" if SECRET_KEYS.search(str(k)) else sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return json.dumps(sanitize(parsed), ensure_ascii=False, indent=2)
        except (ValueError, TypeError):
            pass
        return redact_text(value)
    return value


def readable(value):
    if isinstance(value, str):
        try:
            return json.dumps(json.loads(value), ensure_ascii=False, indent=2)
        except (ValueError, TypeError):
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def atomic_write(path, data):
    fd, tmp = tempfile.mkstemp(prefix=".ronin-export-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sections(detail, status):
    mission = detail.get("mission", {})
    scope = mission.get("scope", {})
    if isinstance(scope, str):
        try:
            scope = json.loads(scope)
        except ValueError:
            scope = {"recorded_scope": scope}
    evidence = detail.get("evidence", [])
    verified = [x for x in evidence if x.get("status") in {"reproduced", "confirmed", "confirmed_lab", "reproduit"}]
    hypotheses = [x for x in evidence if x.get("status") in {"hypothesis", "hypothese", "clue"} or x.get("kind") == "hypothesis"]
    rejected = [x for x in evidence if x.get("status") in {"rejected", "not_reproduced"}]
    observations = [x for x in evidence if x not in verified and x not in hypotheses and x not in rejected]
    engine = status.get("engine", {}) if isinstance(status, dict) else {}
    generation = bool(engine.get("capabilities", {}).get("generation", False))
    engine_text = ("Flash non validé. Aucune génération de GLM-5.3-Flash n'est démontrée par cet export."
                   if not generation else "La disponibilité déclarée du moteur ne prouve pas qu'une génération a été exécutée dans cette mission. Consulter les événements et compteurs enregistrés.")
    engine_text += "\n" + str(engine.get("message", ""))
    result = (f"{len(verified)} résultat(s) marqué(s) reproduit(s) ou confirmé(s) selon les enregistrements de cette mission. "
              "Leur portée reste limitée au protocole et aux conditions consignés."
              if verified else "Aucune vulnérabilité confirmée dans les enregistrements de cette mission.")
    result += f"\n{len(hypotheses)} piste(s) non vérifiée(s), {len(observations)} observation(s), {len(rejected)} piste(s) rejetée(s) ou non reproduite(s)."
    s = [
        ("Périmètre", readable(scope)),
        ("Demande de recherche", mission.get("prompt", "Non enregistrée.")),
        ("Méthode et provenance", "Les observations proviennent exclusivement des outils et événements enregistrés dans la mission. Les analyses statiques par règles produisent des pistes à vérifier. Une seconde réponse du même modèle n'est pas une vérification indépendante.\n" + engine_text),
        ("Résultat", result),
        ("Reproduction et conditions", "Chaque élément ci-dessous fournit sa source, sa date, son statut, son empreinte et la sortie pertinente enregistrée. Les commandes et conditions absentes ne sont pas reconstituées. Les résultats de laboratoire ne démontrent aucun impact sur une cible externe."),
        ("Impact démontré", "L'impact est limité aux observations et reproductions décrites dans les preuves. Aucun montant de prime, niveau de gravité ou impact supplémentaire n'est déduit automatiquement."),
        ("Limites", "Les empreintes attestent la cohérence des artefacts exportés, pas leur vérité. Une observation ancienne n'est pas une validation d'une autre version de cible. Une annulation, une erreur ou une limite de ressources n'est pas une analyse terminée. L'expurgation automatique réduit les secrets courants et emails ; elle ne garantit pas l'absence de toute donnée sensible. Relire le dossier avant toute transmission externe."),
        ("Correction proposée", "Aucune correction n'est présentée comme validée sans modification et contre-test observables. Pour les pistes non reproduites, vérifier d'abord l'atteignabilité, les données contrôlables et les protections effectives, puis consigner le résultat et le contre-test."),
    ]
    for title, rows in [("Résultats reproduits et confirmés", verified), ("Hypothèses et indices non vérifiés", hypotheses), ("Observations", observations), ("Pistes rejetées et non reproduites", rejected)]:
        s.append((title, None))
        if not rows:
            s.append((None, "Aucun élément enregistré."))
        for index, row in enumerate(rows, 1):
            label = STATUS_LABELS.get(row.get("status"), row.get("status", "Non qualifié"))
            text = (f"Statut : {label}\nSource : {row.get('source') or 'Non enregistrée'}\n"
                    f"Date : {row.get('created') or 'Non enregistrée'}\nIdentifiant : {row.get('id', '')}\n"
                    f"SHA-256 du corps original enregistré : {row.get('sha256', 'Absent')}\n\n"
                    + readable(row.get("body", "")))
            s.append((f"{index}. {row.get('title') or row.get('kind') or 'Élément'}", text))
    events = detail.get("events", [])
    s.append(("Activité et décisions enregistrées", "\n\n".join(
        f"{x.get('created', '')} | {x.get('type', '')}\n{readable(x.get('body', ''))}" for x in events) or "Aucun événement enregistré."))
    actions = detail.get("actions", [])
    s.append(("État des actions", readable(actions) if actions else "Aucune action enregistrée."))
    refs = list(dict.fromkeys(str(x["source"]) for x in evidence if x.get("source")))
    s.append(("Références", "\n".join(refs) or "Aucune source enregistrée."))
    return s


def make_pdf(title, subtitle, content):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    regular, bold = "Helvetica", "Helvetica-Bold"
    arial = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    arial_bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if arial.exists() and arial_bold.exists():
        pdfmetrics.registerFont(TTFont("RoninArial", str(arial)))
        pdfmetrics.registerFont(TTFont("RoninArialBold", str(arial_bold)))
        regular, bold = "RoninArial", "RoninArialBold"
    body = ParagraphStyle("Body", fontName=regular, fontSize=9.5, leading=14, textColor=colors.HexColor("#25303c"), spaceAfter=8, splitLongWords=True)
    heading = ParagraphStyle("Heading", parent=body, fontName=bold, fontSize=13, leading=18, textColor=colors.HexColor("#193d5b"), spaceBefore=16, spaceAfter=8, keepWithNext=True)
    title_style = ParagraphStyle("Title", parent=heading, fontSize=24, leading=30, spaceBefore=0, spaceAfter=12)
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=12, textColor=colors.HexColor("#5e6b77"))
    code = ParagraphStyle("Data", parent=body, fontSize=8, leading=12, leftIndent=8, borderPadding=0, backColor=colors.HexColor("#f2f5f7"), spaceAfter=0)
    def clean(value):
        return str(value).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\x00", "")
    def paragraph(value, style=body):
        return Paragraph(escape(clean(value)).replace("\n", "<br/>"), style)
    flow = [paragraph("RONIN", small), paragraph(title, title_style), paragraph(subtitle, small), Spacer(1, 7 * mm)]
    for name, text in content:
        if name:
            flow.append(paragraph(name, heading))
        if text is not None:
            # Keep full evidence in Markdown/JSON; bound PDF excerpts explicitly.
            excerpt = text
            if len(excerpt) > 18000:
                excerpt = excerpt[:18000] + "\n[Extrait PDF limité à 18 000 caractères. Texte complet dans rapport.md et preuves-expurgees.zip.]"
            for block in re.split(r"\n\s*\n", excerpt):
                lines = block.splitlines()
                if len(lines) > 12 or block.lstrip().startswith(("{", "[")):
                    # Individual paragraphs can split across pages without a giant
                    # unbreakable table or code frame.
                    for line in lines:
                        flow.append(paragraph(line or " ", code))
                    flow.append(Spacer(1, 3 * mm))
                else:
                    flow.append(paragraph(block))
    buffer = io.BytesIO()
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d5dce2"))
        canvas.line(19 * mm, 17 * mm, 191 * mm, 17 * mm)
        canvas.setFillColor(colors.HexColor("#5e6b77"))
        canvas.setFont(regular, 8)
        canvas.drawString(19 * mm, 12 * mm, "RONIN | Rapport local | À relire avant transmission")
        canvas.drawRightString(191 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=19 * mm, leftMargin=19 * mm,
                            topMargin=20 * mm, bottomMargin=24 * mm,
                            title=clean(title), author="RONIN", subject="Rapport local de recherche fondé sur les preuves enregistrées")
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def export_report(detail, dest, status):
    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(dest, 0o700)
    safe = sanitize(detail)
    safe_status = sanitize(status)
    mission = safe.get("mission", {})
    title = str(mission.get("title") or "Rapport de mission")
    generated = datetime.now(timezone.utc).isoformat()
    subtitle = f"Mission {mission.get('id', '')} | État : {mission.get('status', 'inconnu')} | Export UTC : {generated}"
    content = sections(safe, safe_status)
    markdown = "# " + title.replace("\n", " ") + "\n\n" + subtitle + "\n\n"
    for heading, text in content:
        if heading:
            markdown += "## " + heading + "\n\n"
        if text is not None:
            # Fences longer than any content fence prevent embedded references
            # from turning into report headings or misleading links.
            if (text.lstrip().startswith(("{", "[")) or "SHA-256 du corps" in text or
                    heading in {"Demande de recherche", "Activité et décisions enregistrées", "Références"}):
                longest = max([len(x) for x in re.findall(r"`+", text)] or [0])
                fence = "`" * max(3, longest + 1)
                markdown += fence + "text\n" + text + "\n" + fence + "\n\n"
            else:
                markdown += text + "\n\n"
    md_path = dest / "rapport.md"
    atomic_write(md_path, markdown.encode())
    manifest = {"schema": 1, "generated_utc": generated, "mission_id": mission.get("id"),
                "redaction": "automatic_patterns_and_sensitive_keys_not_a_guarantee_review_before_sharing",
                "contains_raw_attachments": False, "hashes_attest_integrity_not_truth": True, "files": []}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in [("mission-expurgee.json", safe), ("systeme-expurge.json", safe_status)]:
            data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
            manifest["files"].append({"path": name, "sha256_exported": hashlib.sha256(data).hexdigest()})
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o100600 << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
        for i, row in enumerate(safe.get("evidence", []), 1):
            name = f"preuves/{i:04d}.json"
            data = (json.dumps(row, ensure_ascii=False, indent=2) + "\n").encode()
            manifest["files"].append({"path": name, "sha256_exported": hashlib.sha256(data).hexdigest(),
                                      "sha256_original_record": row.get("sha256")})
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o100600 << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
        info = zipfile.ZipInfo("manifest.json")
        info.external_attr = (0o100600 << 16)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    zip_path = dest / "preuves-expurgees.zip"
    atomic_write(zip_path, archive.getvalue())
    pdf_path = dest / "rapport.pdf"
    errors = []
    try:
        atomic_write(pdf_path, make_pdf(title, subtitle, content))
        pdf_value = str(pdf_path)
    except ImportError as exc:
        pdf_value = None
        errors.append({"code": "PDF_DEPENDENCY_MISSING", "message": "Export PDF indisponible : dépendance reportlab absente.", "detail": redact_text(exc)})
    return {"markdown": str(md_path), "pdf": pdf_value, "evidence": str(zip_path), "errors": errors,
            "redaction_review_required": True, "external_submission": False}
