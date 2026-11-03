import SwiftUI
import Combine
import AppKit
import UniformTypeIdentifiers
import Security

typealias Object = [String: Any]
func s(_ x: Any?) -> String { x.map { String(describing: $0) } ?? "" }
func objects(_ x: Any?) -> [Object] { x as? [Object] ?? [] }
struct Record: Identifiable {
    let value:Object
    init(_ value:Object) {self.value=value}
    var id:String {s(value["id"] ?? value["name"] ?? value["code"] ?? value["sha256"])}
}
func label(_ state: String) -> String {
    ["ready":"Prête", "preparing":"Préparation", "analyzing":"Analyse locale", "waiting_model":"Génération GLM", "paused":"En pause", "resource_paused":"Pause ressources", "stopped":"Arrêtée", "completed":"Terminée", "partial":"Réponse partielle", "error":"Erreur récupérable", "archived":"Archive migrée"][state] ?? state
}

final class Bridge {
    private let queue = DispatchQueue(label: "local.ronin.private-transport")
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var buffered = Data()
    private let lock=NSLock()
    private var callbacks:[String:(Object)->Void]=[:]
    func start() throws {
        guard let resources = Bundle.main.resourceURL else { throw NSError(domain:"RONIN", code:1) }
        let executable = resources.appendingPathComponent("backend/ronin-backend")
        let p = Process(), i = Pipe(), o = Pipe()
        p.executableURL = executable; p.standardInput = i; p.standardOutput = o
        p.standardError = FileHandle.nullDevice
        var env = ProcessInfo.processInfo.environment
        env["PYTHONNOUSERSITE"] = "1"; env["PYTHONDONTWRITEBYTECODE"] = "1"
        p.environment = env
        try p.run(); process = p; input = i.fileHandleForWriting; output = o.fileHandleForReading
        DispatchQueue.global(qos:.userInitiated).async { self.readResponses() }
    }
    private func readResponses() {
        guard let output=output else {return}
        while true {
            let next=output.availableData
            if next.isEmpty { break }
            buffered.append(next)
            while let end=buffered.firstIndex(of:10) {
                let line=buffered[..<end]; buffered.removeSubrange(...end)
                guard let value=try? JSONSerialization.jsonObject(with:line) as? Object, let id=value["id"] as? String else {continue}
                lock.lock(); let callback=callbacks.removeValue(forKey:id); lock.unlock()
                DispatchQueue.main.async {callback?(value)}
            }
        }
        lock.lock(); let pending=Array(callbacks.values); callbacks.removeAll(); lock.unlock()
        DispatchQueue.main.async { for callback in pending {callback(["error":["message":"Le moteur local s’est arrêté. Les preuves sont conservées ; rouvrir RONIN pour reprendre.","code":"BACKEND_STOPPED"]])} }
    }
    func call(_ method: String, _ params: Object = [:], completion: @escaping (Object) -> Void) {
        queue.async {
            let id=UUID().uuidString
            do {
                guard self.process?.isRunning == true, let input = self.input else {
                    throw NSError(domain:"RONIN",code:2,userInfo:[NSLocalizedDescriptionKey:"Le moteur local s’est arrêté. Fermer et rouvrir RONIN pour reprendre les missions conservées."])
                }
                var data = try JSONSerialization.data(withJSONObject:["id":id,"method":method,"params":params])
                self.lock.lock(); self.callbacks[id]=completion; self.lock.unlock()
                data.append(10); try input.write(contentsOf:data)
            } catch {
                self.lock.lock(); self.callbacks.removeValue(forKey:id); self.lock.unlock()
                let message = error.localizedDescription
                DispatchQueue.main.async { completion(["error":["message":message,"code":"BACKEND_STOPPED"]]) }
            }
        }
    }
    func close() {
        // EOF also causes backend shutdown. The parent waits briefly, then terminates only its own child.
        try? input?.close()
        guard let p = process else { return }
        let deadline = Date().addingTimeInterval(7)
        while p.isRunning && Date() < deadline { Thread.sleep(forTimeInterval:0.05) }
        if p.isRunning { p.terminate() }
    }
}

final class AppModel: ObservableObject {
    @Published var missions: [Object] = []
    @Published var selected: String?
    @Published var detail: Object = [:]
    @Published var status: Object = [:]
    @Published var error: String?
    @Published var loading = false
    @Published var skills: [Object] = []
    @Published var lastExport: Object = [:]
    let bridge = Bridge()
    var refreshPending = false
    var backendUnavailable = false
    init() {
        do { try bridge.start(); refresh() } catch { self.error = "Le moteur local n’a pas pu démarrer : \(error.localizedDescription)" }
    }
    func call(_ method: String, _ params: Object = [:], failed: (() -> Void)? = nil, done: ((Any) -> Void)? = nil) {
        bridge.call(method,params) { response in
            if let error = response["error"] as? Object {
                self.error = s(error["message"]) + (s(error["action"]).isEmpty ? "" : "\n"+s(error["action"]))
                self.loading=false; self.refreshPending=false
                if s(error["code"])=="BACKEND_STOPPED" {self.backendUnavailable=true}
                failed?()
            }
            else { done?(response["result"] ?? [:]) }
        }
    }
    func refresh() {
        if refreshPending || backendUnavailable { return }; refreshPending = true
        call("list") { self.missions = objects($0); self.refreshPending = false }
        call("status") { self.status = $0 as? Object ?? [:] }
        if let id = selected { call("detail",["id":id]) { if self.selected==id {self.detail = $0 as? Object ?? [:]} } }
    }
    func select(_ id: String?) {
        selected=id; detail=[:]
        if let id=id {call("detail",["id":id]) {if self.selected==id {self.detail=$0 as? Object ?? [:]}}}
        refresh()
    }
    func create(prompt: String, roots: [String], urls: [String]) {
        loading = true
        call("create",["prompt":prompt,"roots":roots,"urls":urls]) { value in
            self.loading = false
            if let id = (value as? Object)?["id"] as? String {
                self.selected = id
                self.call("start",["id":id]) { _ in self.select(id) }
            }
        }
    }
    func command(_ method: String) { if let id=selected { call(method,["id":id]) { _ in self.refresh() } } }
    func export() {
        guard let id=selected else { return }
        call("export",["id":id]) { value in
            self.lastExport = value as? Object ?? [:]
            let candidate = s(self.lastExport["directory"]).isEmpty ? s(self.lastExport["markdown"]) : s(self.lastExport["directory"])
            if !candidate.isEmpty { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath:candidate)]) }
        }
    }
    var mission: Object { detail["mission"] as? Object ?? [:] }
    var engine: Object { status["engine"] as? Object ?? [:] }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: AppModel?
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) { model?.bridge.close() }
}

@main struct RoninApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject var model = AppModel()
    var body: some Scene {
        WindowGroup("RONIN") {
            ContentView().environmentObject(model).onAppear { delegate.model=model }
                .frame(minWidth:980,minHeight:680)
        }
        .defaultSize(width:1180,height:780)
        .commands { CommandGroup(replacing:.newItem) { Button("Nouvelle mission") { model.select(nil) }.keyboardShortcut("n") } }
        Settings { SettingsView().environmentObject(model).frame(width:620,height:540) }
    }
}

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @State private var settings=false
    @State private var catalog=false
    let timer=Timer.publish(every:1.5,on:.main,in:.common).autoconnect()
    var body: some View {
        NavigationSplitView {
            VStack(alignment:.leading,spacing:20) {
                HStack(spacing:10) {
                    Image(systemName:"square.stack.3d.up").font(.title2).foregroundStyle(Color.accentColor)
                    Text("RONIN").font(.system(size:22,weight:.semibold,design:.rounded))
                }.padding(.top,18)
                Button { model.select(nil) } label: { Label("Nouvelle mission",systemImage:"plus").frame(maxWidth:.infinity,alignment:.leading) }.buttonStyle(.borderedProminent).controlSize(.large)
                Text("Recherches").font(.headline).foregroundStyle(.secondary)
                ScrollView {
                    LazyVStack(spacing:6) {
                        ForEach(model.missions.map {Record($0)}) { record in
                            let mission=record.value
                            Button { model.select(s(mission["id"])) } label: {
                                VStack(alignment:.leading,spacing:5) {
                                    Text(s(mission["title"])).lineLimit(2).font(.system(size:13,weight:.medium)).foregroundStyle(.primary)
                                    Text(label(s(mission["status"]))).font(.caption).foregroundStyle(.secondary)
                                }.frame(maxWidth:.infinity,alignment:.leading).padding(10)
                                    .background(model.selected==s(mission["id"]) ? Color.accentColor.opacity(0.12) : Color.clear,in:RoundedRectangle(cornerRadius:8))
                            }.buttonStyle(.plain)
                        }
                    }
                }
                Spacer(minLength:0)
                Divider()
                Button { catalog=true } label: { Label("Bibliothèque de skills",systemImage:"books.vertical").frame(maxWidth:.infinity,alignment:.leading) }.buttonStyle(.plain).disabled((model.status["catalog"] as? Object)?["available"] as? Bool != true)
                Button { settings=true } label: { Label("Réglages et moteur",systemImage:"slider.horizontal.3").frame(maxWidth:.infinity,alignment:.leading) }.buttonStyle(.plain)
                HStack(alignment:.top,spacing:7) {
                    Image(systemName:"externaldrive.badge.exclamationmark").foregroundStyle(.orange)
                    VStack(alignment:.leading,spacing:3) {
                        Text(model.engine["loaded"] as? Bool == true ? "GLM chargé" : "GLM local sélectionné").font(.caption.weight(.semibold))
                        Text("Calcul et mémoire locaux").font(.caption2).foregroundStyle(.secondary)
                    }
                }.padding(.vertical,10)
            }.padding(.horizontal,18).navigationSplitViewColumnWidth(min:220,ideal:250,max:300)
        } detail: {
            if model.selected==nil { NewMissionView() }
            else { MissionView() }
        }
        .sheet(isPresented:$settings) { SettingsView().environmentObject(model).frame(width:650,height:570) }
        .sheet(isPresented:$catalog) { CatalogView().environmentObject(model).frame(width:860,height:650) }
        .alert("Une action demande ton attention",isPresented:Binding(get:{model.error != nil},set:{if !$0 {model.error=nil}})) { Button("Fermer") { model.error=nil } } message: { Text(model.error ?? "") }
        .onReceive(timer) { _ in
            if model.loading || model.missions.contains(where:{["preparing","analyzing","waiting_model"].contains(s($0["status"]))}) {model.refresh()}
        }
    }
}

struct NewMissionView: View {
    @EnvironmentObject var model: AppModel
    @State private var prompt=""
    @State private var roots:[String]=[]
    @State private var urls:[String]=[]
    @State private var reference=""
    @State private var targeted=false
    func chooseSource() {
        let p=NSOpenPanel(); p.canChooseDirectories=true; p.canChooseFiles=true; p.allowsMultipleSelection=true
        p.message="Choisir les dépôts ou fichiers autorisés pour cette mission."
        if p.runModal() == .OK { for u in p.urls where !roots.contains(u.path) { roots.append(u.path) } }
    }
    var body: some View {
        ScrollView {
            VStack(alignment:.leading,spacing:24) {
                Spacer().frame(height:60)
                Image(systemName:"doc.text.magnifyingglass").font(.system(size:35,weight:.light)).foregroundStyle(Color.accentColor)
                VStack(alignment:.leading,spacing:10) {
                    Text("Une question.\nDes preuves.").font(.system(size:42,weight:.semibold,design:.rounded)).tracking(-1)
                    Text("Décris le système à comprendre et la recherche à mener.").font(.title3).foregroundStyle(.secondary)
                }
                VStack(alignment:.leading,spacing:12) {
                    TextEditor(text:$prompt).font(.system(size:16)).scrollContentBackground(.hidden).frame(minHeight:125)
                        .accessibilityLabel("Décrire la mission")
                    Divider()
                    HStack {
                        Button(action:chooseSource) { Label("Ajouter des sources",systemImage:"paperclip") }.buttonStyle(.borderless)
                        Spacer()
                        Text("Dépôt, fichier ou URL").font(.caption).foregroundStyle(.secondary)
                    }
                    ForEach(roots,id:\.self) { path in
                        HStack { Image(systemName:"folder"); Text(path).lineLimit(1).truncationMode(.middle); Spacer(); Button { roots.removeAll{$0==path} } label:{Image(systemName:"xmark.circle.fill")} .buttonStyle(.plain).accessibilityLabel("Retirer \(URL(fileURLWithPath:path).lastPathComponent)") }.font(.caption)
                    }
                    HStack {
                        TextField("URL de référence (n’accorde aucune autorisation)",text:$reference).textFieldStyle(.plain).accessibilityLabel("URL de référence")
                        Button("Ajouter") { if URL(string:reference)?.scheme != nil { urls.append(reference); reference="" } }.disabled(reference.isEmpty)
                    }
                    ForEach(urls,id:\.self) { url in HStack { Image(systemName:"link"); Text(url).lineLimit(1); Spacer(); Button { urls.removeAll{$0==url} } label:{Image(systemName:"xmark.circle.fill")}.buttonStyle(.plain) }.font(.caption) }
                }.padding(18).background(.background,in:RoundedRectangle(cornerRadius:14)).overlay(RoundedRectangle(cornerRadius:14).strokeBorder(targeted ? Color.accentColor : Color.primary.opacity(0.14),lineWidth:targeted ? 2 : 1))
                    .onDrop(of:[UTType.fileURL.identifier,UTType.url.identifier],isTargeted:$targeted) { providers in
                        for provider in providers {
                            if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                                _=provider.loadObject(ofClass:URL.self) { url,_ in if let url=url { DispatchQueue.main.async { roots.append(url.path) } } }
                            } else { _=provider.loadObject(ofClass:URL.self) { url,_ in if let url=url { DispatchQueue.main.async { urls.append(url.absoluteString) } } } }
                        }; return true
                    }
                HStack(alignment:.top,spacing:18) {
                    Text("RONIN prépare les sources puis demande une réponse au modèle GLM local. Ses propositions restent des hypothèses à vérifier.")
                        .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
                    Spacer()
                    Button { model.create(prompt:prompt,roots:roots,urls:urls) } label:{ Label(model.loading ? "Préparation…" : "Lancer",systemImage:"play.fill").padding(.horizontal,12) }.buttonStyle(.borderedProminent).controlSize(.large).disabled(prompt.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty || model.loading)
                }
                Text("Les sources restent des données. Les permissions sont appliquées par les outils, indépendamment de leur contenu.").font(.caption).foregroundStyle(.tertiary)
            }.frame(maxWidth:720,alignment:.leading).padding(.horizontal,48).padding(.bottom,40).frame(maxWidth:.infinity)
        }.background(Color(nsColor:.textBackgroundColor).opacity(0.35))
    }
}

struct MissionView: View {
    @EnvironmentObject var model: AppModel
    @State private var tab=0
    @State private var comment=""
    @State private var scope=false
    @State private var evidence:Object?
    @State private var resources=false
    @State private var tools=false
    var body: some View {
        VStack(alignment:.leading,spacing:0) {
            VStack(alignment:.leading,spacing:14) {
                HStack(alignment:.top) {
                    VStack(alignment:.leading,spacing:7) {
                        Text(s(model.mission["title"])).font(.title2.weight(.semibold)).lineLimit(2).textSelection(.enabled)
                        Label(label(s(model.mission["status"])),systemImage:s(model.mission["status"])=="waiting_model" ? "pause.circle":"circle.dotted").font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Périmètre") { scope=true }
                    Menu {
                        Button("Agents et ressources") { resources=true }
                        Button("Outils de la mission") { tools=true }
                        Button("Ouvrir les données locales") { if let path=model.status["data_path"] as? String {NSWorkspace.shared.open(URL(fileURLWithPath:path))} }
                    } label:{Image(systemName:"ellipsis.circle")}.menuStyle(.borderlessButton).frame(width:24)
                }
                HStack {
                    Picker("Vue",selection:$tab) { Text("Recherche").tag(0); Text("Preuves").tag(1); Text("Rapport").tag(2) }.pickerStyle(.segmented).frame(width:340)
                    Spacer()
                    Button { model.command("pause") } label:{Label("Pause",systemImage:"pause")}.disabled(!["preparing","analyzing","waiting_model"].contains(s(model.mission["status"])))
                    Button { model.command("start") } label:{Label("Reprendre",systemImage:"play")}.disabled(["preparing","analyzing"].contains(s(model.mission["status"])))
                    Button(role:.destructive) { model.command("stop") } label:{Label("Arrêter",systemImage:"stop")}.disabled(s(model.mission["status"])=="stopped")
                }
            }.padding(24)
            Divider()
            if tab==0 {
                ScrollView {
                    VStack(alignment:.leading,spacing:18) {
                        Text(s(model.mission["prompt"])).font(.title3).textSelection(.enabled).padding(.bottom,8)
                        ForEach(objects(model.detail["events"]).map {Record($0)}) { record in
                            let event=record.value
                            HStack(alignment:.top,spacing:12) {
                                Image(systemName:s(event["type"])=="model_blocked" ? "pause.circle" : s(event["type"])=="user" ? "person.crop.circle" : "doc.text").foregroundStyle(s(event["type"])=="model_blocked" ? Color.orange : Color.secondary).frame(width:20)
                                VStack(alignment:.leading,spacing:5) {
                                    Text(eventTitle(s(event["type"]))).font(.callout.weight(.semibold))
                                    Text(eventBody(event)).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                                    Text(String(s(event["created"]).prefix(19)).replacingOccurrences(of:"T",with:" ")+" UTC").font(.caption2).foregroundStyle(.tertiary)
                                }
                            }.frame(maxWidth:.infinity,alignment:.leading)
                        }
                    }.padding(28).frame(maxWidth:850,alignment:.leading).frame(maxWidth:.infinity,alignment:.leading)
                }
                Divider()
                HStack(alignment:.bottom) {
                    TextField("Ajouter une observation à la mission",text:$comment,axis:.vertical).lineLimit(1...4).textFieldStyle(.plain).accessibilityLabel("Observation de mission")
                    Button("Enregistrer") {
                        if let id=model.selected { model.call("comment",["id":id,"text":comment]) { _ in comment=""; model.refresh() } }
                    }.disabled(comment.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
                }.padding(20)
            } else if tab==1 {
                ScrollView {
                    VStack(alignment:.leading,spacing:14) {
                        if objects(model.detail["evidence"]).isEmpty { Text("Les observations des outils apparaîtront ici, avec leur source et leur date.").foregroundStyle(.secondary).padding(.top,20) }
                        ForEach(objects(model.detail["evidence"]).map {Record($0)}) { record in
                            let item=record.value
                            Button { evidence=item } label:{
                                HStack(alignment:.top,spacing:14) {
                                    Image(systemName:s(item["kind"])=="hypothesis" ? "questionmark.diamond" : "doc.text.magnifyingglass").font(.title2).foregroundStyle(Color.accentColor)
                                    VStack(alignment:.leading,spacing:5) {
                                        Text(s(item["title"])).font(.headline).foregroundStyle(.primary)
                                        Text(s(item["source"])).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                                        Text(s(item["status"])=="hypothesis" ? "Hypothèse à vérifier" : s(item["status"])=="reproduced_lab_only" ? "Reproduit dans le laboratoire uniquement" : "Observation enregistrée").font(.caption).foregroundStyle(.secondary)
                                    }; Spacer(); Image(systemName:"chevron.right").foregroundStyle(.secondary)
                                }.padding(16).background(Color(nsColor:.controlBackgroundColor),in:RoundedRectangle(cornerRadius:10))
                            }.buttonStyle(.plain)
                        }
                        HStack {
                            Button("Reproduction contrôlée") { if let id=model.selected,let sc=model.mission["scope"] as? Object,let root=(sc["roots"] as? [String])?.first { model.call("tool",["id":id,"tool":"lab.verify","args":["path":root]]) { _ in model.refresh() } } }.disabled(((model.mission["scope"] as? Object)?["roots"] as? [String] ?? []).isEmpty)
                            Text("Teste une primitive connue ; ne confirme pas une vulnérabilité de la cible.").font(.caption).foregroundStyle(.secondary)
                        }.padding(.top,12)
                    }.padding(26)
                }
            } else {
                ScrollView {
                    VStack(alignment:.leading,spacing:22) {
                        Image(systemName:"doc.richtext").font(.system(size:38,weight:.light)).foregroundStyle(Color.accentColor)
                        Text("Un rapport qui dit ce qui a été observé.").font(.title.weight(.semibold))
                        Text("Le rapport rassemble le périmètre, les observations datées, les conditions de reproduction et les limites. Les hypothèses restent séparées des résultats vérifiés.").font(.body).foregroundStyle(.secondary)
                        Label("Les réponses du modèle restent à vérifier",systemImage:"checkmark.shield").foregroundStyle(.secondary)
                        Button { model.export() } label:{Label("Exporter Markdown, PDF et preuves",systemImage:"square.and.arrow.up")}.buttonStyle(.borderedProminent).controlSize(.large)
                        if !model.lastExport.isEmpty { Text("Export conservé dans les données locales de RONIN.").font(.caption).foregroundStyle(.secondary) }
                        Divider()
                        Text("Aucun envoi externe n’est effectué. Les contenus détectés comme secrets sont expurgés ; une relecture du dossier exporté reste nécessaire avant sa diffusion.").font(.callout).foregroundStyle(.secondary)
                    }.frame(maxWidth:630,alignment:.leading).padding(40).frame(maxWidth:.infinity,alignment:.leading)
                }
            }
        }
        .sheet(isPresented:$scope) { ScopeView().environmentObject(model).frame(width:650,height:590) }
        .sheet(isPresented:$resources) { ResourcesView().environmentObject(model).frame(width:680,height:540) }
        .sheet(isPresented:$tools) { ToolsView().environmentObject(model).frame(width:760,height:660) }
        .sheet(isPresented:Binding(get:{evidence != nil},set:{if !$0 {evidence=nil}})) {
            VStack(alignment:.leading,spacing:16) {
                HStack { Text(s(evidence?["title"])).font(.title2); Spacer(); Button("Fermer") { evidence=nil } }
                Text(s(evidence?["source"])).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                ScrollView { Text(s(evidence?["body"])).font(.system(size:12,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading) }
                Text("SHA-256 du contenu expurgé : "+s(evidence?["sha256"])).font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
            }.padding(24).frame(width:800,height:600)
        }
    }
    func eventTitle(_ kind:String)->String { ["decision":"Décision enregistrée","scope":"Périmètre modifié","model_blocked":"Moteur indisponible","user":"Ton observation","stop":"Arrêt","pause":"Pause","error":"Erreur récupérable","tool_error":"Outil interrompu","legacy_import":"Archive conservée"][kind] ?? kind }
    func eventBody(_ event:Object)->String {
        if s(event["type"])=="model_blocked" { return "Le moteur local est actuellement indisponible. La préparation et les preuves locales sont conservées. Consulte les réglages du moteur pour connaître le blocage observé." }
        return s(event["body"])
    }
}

struct ScopeView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) var dismiss
    @State private var roots=""
    @State private var origins=""
    @State private var authorized=false
    var body: some View {
        VStack(alignment:.leading,spacing:18) {
            Text("Périmètre de la mission").font(.title2.weight(.semibold))
            Text("Les modifications s’appliquent aux actions futures. Une restriction est appliquée immédiatement ; les actions en cours restent visibles dans les preuves.").font(.callout).foregroundStyle(.secondary)
            Text("Dépôts et fichiers autorisés, un chemin par ligne").font(.headline)
            TextEditor(text:$roots).font(.system(size:12,design:.monospaced)).frame(height:100).border(Color.secondary.opacity(0.25)).accessibilityLabel("Chemins du périmètre")
            Text("Origines HTTP autorisées, une par ligne").font(.headline)
            TextEditor(text:$origins).font(.system(size:12,design:.monospaced)).frame(height:75).border(Color.secondary.opacity(0.25)).accessibilityLabel("Origines réseau du périmètre")
            Toggle("J’autorise les requêtes GET vers les origines indiquées",isOn:$authorized)
            Text("Aucune redirection n’est suivie. Aucun cookie ni identifiant n’est envoyé. Les scripts de skill ne reçoivent aucune permission.").font(.caption).foregroundStyle(.secondary)
            Spacer()
            HStack { Button("Annuler") { dismiss() }; Spacer(); Button("Enregistrer le périmètre") { if let id=model.selected {
                model.call("scope",["id":id,"roots":roots.split(separator:"\n").map(String.init),"origins":origins.split(separator:"\n").map(String.init),"authorized":authorized]) { _ in model.refresh(); dismiss() }
            } }.buttonStyle(.borderedProminent) }
        }.padding(26).onAppear {
            let scope=model.mission["scope"] as? Object ?? [:]
            roots=(scope["roots"] as? [String] ?? []).joined(separator:"\n")
            origins=(scope["origins"] as? [String] ?? []).joined(separator:"\n")
            authorized=scope["active_network_authorized"] as? Bool ?? false
        }
    }
}

struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) var dismiss
    @State private var path=""
    var body: some View {
        VStack(alignment:.leading,spacing:20) {
            HStack { Text("Moteur et stockage").font(.title2.weight(.semibold)); Spacer(); Button("Fermer") { dismiss() } }
            Label("GLM-5.3 Flash Q2_K_S local",systemImage:"externaldrive").font(.headline)
            Text("RONIN utilise le modèle GLM local avec le moteur natif partagé d’AirLLM Next. Les garde-fous mémoire restent actifs pendant la génération.").font(.body).foregroundStyle(.secondary)
            Text("Dossier des poids").font(.headline)
            HStack {
                TextField("Sélectionner un checkpoint complet",text:$path).textFieldStyle(.roundedBorder).accessibilityLabel("Chemin du modèle")
                Button("Choisir…") { let panel=NSOpenPanel(); panel.canChooseDirectories=true; panel.canChooseFiles=false; if panel.runModal() == .OK {path=panel.url!.path} }
                Button("Enregistrer") { model.call("settings",["model_path":path]) { _ in model.refresh() } }
            }
            HStack {
                Button("Charger GLM") { model.call("engine_load") { _ in model.refresh() } }
                Button("Décharger") { model.call("engine_unload") { _ in model.refresh() } }
                Text(model.engine["loaded"] as? Bool == true ? "Chargé" : "Non chargé").foregroundStyle(.secondary)
            }
            Text("Aucun téléchargement automatique. Un dossier nommé SSD ne prouve pas qu’un disque externe est connecté.").font(.caption).foregroundStyle(.secondary)
            Divider()
            ScrollView {
                VStack(alignment:.leading,spacing:12) {
                    Text("Blocages observés").font(.headline)
                    ForEach(objects(model.engine["blockers"]).map {Record($0)}) { record in
                            let item=record.value
                        VStack(alignment:.leading,spacing:4) { Text(s(item["message"])).font(.callout); Text(s(item["action"])).font(.caption).foregroundStyle(.secondary) }
                    }
                    Text("Les performances et la fidélité mesurées sont décrites dans le rapport de validation livré avec cette version.").font(.callout).foregroundStyle(.secondary)
                    Text("Données : "+s(model.status["data_path"])).font(.caption).textSelection(.enabled)
                    Text("L’interface communique par un canal privé. Le service d’inférence écoute uniquement sur cette machine et s’arrête à la fermeture de l’application.").font(.caption).foregroundStyle(.secondary)
                }.frame(maxWidth:.infinity,alignment:.leading)
            }
        }.padding(26).onAppear { path=s(model.status["model_path"]) }
    }
}

struct CatalogView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) var dismiss
    @State private var query=""
    @State private var document=""
    @State private var chosen=""
    @State private var attached=false
    var body: some View {
        VStack(alignment:.leading,spacing:16) {
            HStack { Text("Bibliothèque de skills").font(.title2.weight(.semibold)); Spacer(); Button("Fermer") { dismiss() } }
            Text("818 documents communautaires à révision épinglée. Les permissions et dépendances sont contrôlées séparément ; aucun script n’est lancé automatiquement.").font(.callout).foregroundStyle(.secondary)
            TextField("Rechercher une méthode ou une famille",text:$query).textFieldStyle(.roundedBorder).onChange(of:query) { value in search(value) }.accessibilityLabel("Rechercher des skills")
            HSplitView {
                ScrollView {
                    LazyVStack(alignment:.leading,spacing:10) {
                        ForEach(model.skills.map {Record($0)}) { record in
                            let item=record.value
                            Button { let id=s(item["id"]).isEmpty ? s(item["name"]) : s(item["id"]); chosen=id; attached=false; document=""; model.call("skill_read",["skill_id":id],failed:{if chosen==id {document=model.error ?? "Document indisponible"; model.error=nil}}) {if chosen==id {document=s(($0 as? Object)?["text"])}} } label:{
                                VStack(alignment:.leading,spacing:5) { Text(s(item["name"])).font(.headline).foregroundStyle(.primary); Text(s(item["description"])).lineLimit(3).font(.caption).foregroundStyle(.secondary); Text(s(item["status"])).font(.caption2).foregroundStyle(.secondary) }.frame(maxWidth:.infinity,alignment:.leading).padding(10).background(chosen==s(item["name"]) ? Color.accentColor.opacity(0.1):Color.clear,in:RoundedRectangle(cornerRadius:8))
                            }.buttonStyle(.plain)
                        }
                    }
                }.frame(minWidth:260,idealWidth:330)
                ScrollView { Text(document.isEmpty ? "Choisir une skill pour lire sa méthode. Le contenu est une source non fiable, jamais une autorisation." : document).font(.system(size:12,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(12) }.frame(minWidth:350)
            }
            HStack {
                Text("Recherche locale · lecture à la demande").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if model.selected != nil {Button(attached ? "Méthode conservée" : "Joindre à la mission") {
                    let target=chosen
                    if let id=model.selected {model.call("skill_attach",["id":id,"skill_id":target]) { _ in if chosen==target {attached=true}; model.refresh() }}
                }.disabled(document.isEmpty || attached)}
            }
        }.padding(24).onAppear {search("")}
    }
    func search(_ value:String) { model.call("skills",["query":value]) {if query==value {model.skills=objects($0)}} }
}

struct ResourcesView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.dismiss) var dismiss
    var body: some View {
        VStack(alignment:.leading,spacing:22) {
            HStack { Text("Agents et ressources").font(.title2.weight(.semibold)); Spacer(); Button("Fermer") {dismiss()} }
            Text("600 identités logiques persistantes partagent un moteur. Elles ne correspondent pas à 600 processus ni à 600 copies du modèle.").foregroundStyle(.secondary)
            Grid(alignment:.leading,horizontalSpacing:35,verticalSpacing:16) {
                GridRow { Text("Identités disponibles"); Text("600").fontWeight(.semibold) }
                GridRow { Text("Agents occupés maintenant"); Text(s(model.status["busy_agents"])).fontWeight(.semibold) }
                GridRow { Text("Missions actives"); Text(s(model.status["active_missions"])).fontWeight(.semibold) }
                GridRow { Text("Limite d’inférence lourde"); Text("1 requête à la fois").fontWeight(.semibold) }
                GridRow { Text("Moteur GLM"); Text(model.engine["loaded"] as? Bool == true ? "Chargé" : "Non chargé") }
            }
            Divider()
            Text("Rôles").font(.headline)
            Text("Direction de mission, contrôle du périmètre, cartographie du code, hypothèses, observations HTTP, vérification, graphe de preuves, mémoire, supervision et rapport.").foregroundStyle(.secondary)
            Spacer()
        }.padding(28)
    }
}

struct ToolsView: View {
    @EnvironmentObject var model:AppModel
    @Environment(\.dismiss) var dismiss
    @State private var path=""
    @State private var query=""
    @State private var url=""
    @State private var result=""
    @State private var busy=false
    func invoke(_ tool:String,_ args:Object) {
        guard let id=model.selected else {return}
        busy=true
        model.call("tool",["id":id,"tool":tool,"args":args],failed:{busy=false; result=model.error ?? "Outil interrompu"; model.error=nil}) { value in
            busy=false
            if let data=try? JSONSerialization.data(withJSONObject:value,options:[.prettyPrinted,.sortedKeys]),let text=String(data:data,encoding:.utf8) {result=text}
            model.refresh()
        }
    }
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            HStack {Text("Outils de la mission").font(.title2.weight(.semibold)); Spacer(); Button("Fermer") {dismiss()} }
            Text("Chaque action est contrôlée par le périmètre courant et conserve une preuve. Aucun script de dépôt ou de skill n’est exécuté.").font(.callout).foregroundStyle(.secondary)
            TextField("Chemin autorisé",text:$path).textFieldStyle(.roundedBorder).accessibilityLabel("Chemin pour les outils")
            HStack {
                Button("Lire le fichier") {invoke("code.read",["path":path])}.disabled(busy || path.isEmpty)
                Button("Cartographier") {invoke("code.list",["path":path])}.disabled(busy || path.isEmpty)
                Button("Analyser Python") {invoke("static.python",["path":path])}.disabled(busy || path.isEmpty)
                Button("Compter les lignes") {invoke("terminal.local",["path":URL(fileURLWithPath:path).deletingLastPathComponent().path,"argv":["/usr/bin/wc","-l",path]])}.disabled(busy || path.isEmpty)
            }
            HStack {
                TextField("Texte à rechercher dans les sources",text:$query).textFieldStyle(.roundedBorder).accessibilityLabel("Recherche dans le code")
                Button("Rechercher") {invoke("code.search",["path":path,"query":query])}.disabled(busy || path.isEmpty || query.isEmpty)
            }
            Divider()
            HStack {
                TextField("URL autorisée pour une requête GET",text:$url).textFieldStyle(.roundedBorder).accessibilityLabel("URL de requête HTTP")
                Button("Observer HTTP") {invoke("http.get",["url":url])}.disabled(busy || url.isEmpty)
            }
            Text("Les origines réseau doivent avoir été autorisées dans Périmètre. Les redirections sont enregistrées sans être suivies.").font(.caption).foregroundStyle(.secondary)
            ScrollView {Text(result.isEmpty ? "Les sorties réelles de l’outil apparaîtront ici." : result).font(.system(size:12,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)}.padding(12).background(Color(nsColor:.textBackgroundColor),in:RoundedRectangle(cornerRadius:8))
            HStack {Text(busy ? "Outil en cours…" : "Sorties et preuves locales").font(.caption).foregroundStyle(.secondary); Spacer(); Button("Arrêter la mission") {model.command("stop"); busy=false}.foregroundStyle(.red) }
        }.padding(26).onAppear {path=((model.mission["scope"] as? Object)?["roots"] as? [String])?.first ?? ""}
    }
}
