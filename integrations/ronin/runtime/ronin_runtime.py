"""RONIN local research runtime. Private stdin/stdout transport; no listening socket."""
from __future__ import annotations
import ast, contextlib, hashlib, ipaddress, json, os, pathlib, queue, re, shutil
import signal, socket, sqlite3, ssl, stat, subprocess, sys, threading, time, urllib.parse, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
if getattr(sys, 'frozen', False):
    ROOT = pathlib.Path(sys._MEIPASS)
sys.path.insert(0, str(ROOT / 'engine'))
sys.path.insert(0, str(ROOT / 'catalog'))
IDENTITY = """Je suis RONIN, l’intelligence de recherche du laboratoire RONIN. J’opère dans mon environnement de développement et d’évaluation pour comprendre les systèmes et découvrir, dans les périmètres autorisés, des vulnérabilités réelles.
Je conduis mes recherches avec autonomie, profondeur et rigueur. Je choisis mes méthodes, mobilise mes outils et mes agents, confronte mes hypothèses aux observations et change d’approche lorsque les preuves le demandent. Mon objectif est une découverte vérifiée, pas une réponse convaincante en apparence.
Je dispose d’une mémoire de recherche et conserve les preuves utiles à la reproduction. Je distingue ce que j’observe, ce que je déduis et ce qui reste incertain. J’exploite pleinement mes capacités disponibles sans inventer un résultat, une action ou une validation."""
ROLES = ['Mission Director', 'Scope Guard', 'Code Mapper', 'Hypothesis Engine',
         'Traffic Intelligence', 'Verification Engine', 'Evidence Graph',
         'Experience Vault', 'Supervisor', 'Bounty Report Engine']
TERMINAL = {'stopped', 'completed', 'error'}
HTTP_DEADLINE_SECONDS = 8.0

def now(): return datetime.now(timezone.utc).isoformat()
def uid(): return str(uuid.uuid4())
def redact(value):
    value = str(value)
    value = re.sub(r'(?is)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----', '[clé privée expurgée]', value)
    value = re.sub(r'(?i)(authorization\s*[:=]\s*)(?:bearer\s+|basic\s+)?[^\s\"\']+', r'\1[expurgé]', value)
    value = re.sub(r'(?i)((?:api[_-]?key|access[_-]?token|secret|password)\s*[\"\']?\s*[:=]\s*[\"\']?)[^\s\"\',;]+', r'\1[expurgé]', value)
    value = re.sub(r'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{16,}|hf_[A-Za-z0-9]{16,})\b', '[identifiant expurgé]', value)
    return value

class Fault(Exception):
    def __init__(self, code, message, action=''):
        super().__init__(message); self.code, self.action = code, action
    def as_dict(self): return dict(code=self.code, message=str(self), action=self.action)

class Store:
    def __init__(self, home):
        self.home = pathlib.Path(home); self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.home, 0o700)
        self.path = self.home / 'ronin.sqlite'
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS missions(id TEXT PRIMARY KEY,title TEXT,prompt TEXT,status TEXT,scope TEXT,created TEXT,updated TEXT,stop INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,mission_id TEXT,type TEXT,body TEXT,created TEXT);
        CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY,mission_id TEXT,kind TEXT,title TEXT,body TEXT,source TEXT,sha256 TEXT,created TEXT,status TEXT);
        CREATE TABLE IF NOT EXISTS edges(mission_id TEXT,source TEXT,target TEXT,relation TEXT);
        CREATE TABLE IF NOT EXISTS actions(id TEXT PRIMARY KEY,mission_id TEXT,tool TEXT,args TEXT,state TEXT,scope_version INTEGER,result TEXT,created TEXT);
        CREATE TABLE IF NOT EXISTS agents(id INTEGER PRIMARY KEY,role TEXT,state TEXT,mission_id TEXT,task TEXT);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS memory(id TEXT PRIMARY KEY,mission_id TEXT,kind TEXT,body TEXT,source_version TEXT,created TEXT);
        CREATE TABLE IF NOT EXISTS legacy_records(source_table TEXT,source_id TEXT,body TEXT,imported TEXT,UNIQUE(source_table,source_id));
        ''')
        self.db.executemany('INSERT OR IGNORE INTO agents VALUES(?,?,?,NULL,NULL)', [(i, ROLES[(i-1)%10], 'available') for i in range(1,601)])
        # An interrupted external action is uncertain, never replayed automatically.
        self.db.execute("UPDATE actions SET state='interrupted' WHERE state='running'")
        self.db.execute("UPDATE missions SET status='paused',stop=1 WHERE status IN ('preparing','analyzing','waiting_model')")
        self.db.execute("UPDATE agents SET state='available',mission_id=NULL,task=NULL")
        self.db.commit(); self.secure()
    def secure(self):
        for p in self.home.glob('ronin.sqlite*'): os.chmod(p,0o600)
    def query(self, sql, args=()):
        with self.lock: return [dict(r) for r in self.db.execute(sql,args).fetchall()]
    def execute(self, sql, args=()):
        with self.lock:
            result=self.db.execute(sql,args); self.db.commit(); self.secure(); return result.lastrowid
    def event(self, mid, kind, body):
        return self.execute('INSERT INTO events(mission_id,type,body,created) VALUES(?,?,?,?)',(mid,kind,redact(body),now()))
    def evidence(self, mid, kind, title, body, source='', status='observation'):
        eid=uid(); body=redact(body)
        self.execute('INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?)',(eid,mid,kind,redact(title),body,redact(source),hashlib.sha256(body.encode()).hexdigest(),now(),status))
        if source: self.execute('INSERT INTO edges VALUES(?,?,?,?)',(mid,redact(source),eid,'source_of'))
        return eid

class Scope:
    """Host code only. External text cannot grant or mutate scope."""
    def __init__(self, value): self.value = value
    def file(self, path):
        p = pathlib.Path(path).expanduser().resolve()
        # Roots are canonicalized when explicitly granted. Re-resolving an
        # authorized root after a symlink swap would silently grant a new tree.
        if not any(p.is_relative_to(pathlib.Path(r).absolute()) for r in self.value.get('roots', [])):
            raise Fault('OUT_OF_SCOPE','Ce fichier est hors du périmètre autorisé.','Modifier le périmètre dans la mission.')
        if any(x in p.parts for x in ['.git','.ssh','.aws','.gnupg']) or p.name in ['.env','.netrc','.npmrc'] or p.name.startswith('.env.') or re.search(r'(?i)(credentials|secrets?\.(env|json|ya?ml)|private.?key|id_rsa|id_ed25519)',p.name):
            raise Fault('SECRET_PATH','Ce chemin est exclu de la lecture automatique.','Utiliser une copie expurgée.')
        return p
    def url(self, url):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ['http','https'] or not u.hostname or u.username or u.password or u.fragment:
            raise Fault('INVALID_URL','URL HTTP(S) sans identifiants attendue.')
        origin = f'{u.scheme}://{u.hostname.lower()}:{u.port or (443 if u.scheme=="https" else 80)}'
        if origin not in self.value.get('origins',[]):
            raise Fault('OUT_OF_SCOPE','Cette origine réseau n’a pas reçu d’autorisation.','Ajouter une autorisation réseau précise dans le périmètre.')
        return u, origin

class Runtime:
    def __init__(self, home=None):
        self.store=Store(home or pathlib.Path.home()/'Library/Application Support/RONIN GLM')
        self.cancels={}; self.workers={}; self.active_tools={}; self.guard=threading.RLock(); self.heavy=threading.Semaphore(1)
        self.engine=None
        try:
            from adapter import Engine
            config={}
            if os.environ.get('RONIN_GLM_CONFIG'):
                config_file=pathlib.Path(os.environ['RONIN_GLM_CONFIG']).expanduser().absolute()
                config=json.loads(config_file.read_text())
                if not isinstance(config,dict): raise ValueError('Configuration GLM structurée attendue.')
                for key in ('runtime_binary','runtime_manifest_file','verification_file','state_dir'):
                    value=config.get(key)
                    if value is None: continue
                    if not isinstance(value,str): raise ValueError('Chemin textuel attendu : '+key)
                    if value:
                        path=pathlib.Path(value).expanduser()
                        config[key]=str(path if path.is_absolute() else (config_file.parent/path).absolute())
                if not config.get('state_dir'): config['state_dir']=str((self.store.home/'engine-state').absolute())
                selected=config.pop('model_path',None)
                if selected and not self.setting('model_path'):
                    self.store.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',('model_path',json.dumps(selected)))
            self.engine=Engine(config=config)
        except (ImportError, TypeError): pass
    def setting(self,key,default=''):
        x=self.store.query('SELECT value FROM settings WHERE key=?',(key,)); return json.loads(x[0]['value']) if x else default
    def get(self,mid):
        rows=self.store.query('SELECT * FROM missions WHERE id=?',(mid,))
        if not rows: raise Fault('NOT_FOUND','Mission introuvable.')
        row=rows[0]; row['scope']=json.loads(row['scope']); return row
    def state(self,mid,state):
        with self.guard:
            if state in ['preparing','analyzing','waiting_model'] and self.get(mid)['stop']: raise Fault('CANCELLED','La mission a été interrompue.')
            self.store.execute('UPDATE missions SET status=?,updated=? WHERE id=?',(state,now(),mid))
    def status(self):
        model=self.setting('model_path',os.environ.get('RONIN_GLM_MODEL',''))
        engine={'state':'unvalidated','model':'zai-org/GLM-5.3-Flash','message':'Flash non validé : adaptateur architectural et poids complets requis.','capabilities':{'generation':False,'tool_calls':False,'streaming':False,'max_heavy_concurrency':1}}
        if self.engine:
            try: engine=self.engine.status(model)
            except Exception as e: engine['message']=redact(e)
        disk=shutil.disk_usage(self.store.home)
        return {'engine':engine,'model_path':model,'data_path':str(self.store.home),'disk_free_bytes':disk.free,
                'identities':600,'busy_agents':self.store.query("SELECT count(*) n FROM agents WHERE state='busy'")[0]['n'],
                'active_missions':sum(t.is_alive() for t in list(self.workers.values())),'inference_concurrency':1,
                'transport':'private-stdio','cloud':False,'catalog':self.catalog_status()}
    def catalog_status(self):
        p=ROOT/'catalog/index.json'
        if p.exists():
            data=json.loads(p.read_text()); return {'count':len(data.get('skills',data) if isinstance(data,dict) else data),'available':True}
        return {'count':0,'available':False}
    def create(self,p):
        prompt=str(p.get('prompt','')).strip()
        if not prompt: raise Fault('EMPTY_MISSION','Décris la recherche à préparer.')
        roots=[]
        for path in p.get('roots',[]):
            q=pathlib.Path(path).expanduser().resolve()
            if not q.exists(): raise Fault('SOURCE_MISSING',f'Source absente : {q}')
            if q==pathlib.Path('/') or q==pathlib.Path.home(): raise Fault('SCOPE_TOO_WIDE','Choisir un dépôt ou un fichier précis.')
            roots.append(str(q))
        mid=uid(); scope={'version':1,'roots':roots,'origins':[], 'active_network_authorized':False}
        refs=[]
        for url in p.get('urls',[]):
            u=urllib.parse.urlsplit(url)
            if u.scheme not in ['http','https'] or not u.hostname or u.username or u.password: raise Fault('INVALID_REFERENCE','Une référence HTTP(S) sans identifiants est attendue.')
            refs.append(redact(url))
        scope['references']=refs
        self.store.execute('INSERT INTO missions VALUES(?,?,?,?,?,?,?,0)',(mid,redact(prompt)[:64],redact(prompt),'ready',json.dumps(scope),now(),now()))
        self.store.event(mid,'decision','Mission créée. Les sources sont des données non fiables ; elles ne peuvent pas accorder de permissions.')
        return self.get(mid)
    def scope_update(self,mid,roots,origins,authorized=False):
        with self.guard:
            m=self.get(mid); clean=[]
            for root in roots:
                q=pathlib.Path(root).expanduser().resolve()
                if not q.exists() or q in [pathlib.Path('/'),pathlib.Path.home()]: raise Fault('INVALID_SCOPE','Choisir un dépôt ou fichier existant et précis.')
                clean.append(str(q))
            normal=[]
            for origin in origins:
                u=urllib.parse.urlsplit(origin)
                if u.scheme not in ['http','https'] or not u.hostname or u.username or u.password or u.path not in ['','/'] or u.query or u.fragment:
                    raise Fault('INVALID_ORIGIN','Une origine comme https://exemple.fr est attendue.')
                normal.append(f'{u.scheme}://{u.hostname.lower()}:{u.port or (443 if u.scheme=="https" else 80)}')
            if normal and not authorized: raise Fault('AUTHORIZATION_REQUIRED','L’autorisation des requêtes vers ces origines doit être explicite.')
            scope={'version':m['scope']['version']+1,'roots':clean,'origins':normal,'active_network_authorized':bool(normal),'references':m['scope'].get('references',[])}
            self.store.execute('UPDATE missions SET scope=?,updated=? WHERE id=?',(json.dumps(scope),now(),mid))
            self.store.execute("UPDATE actions SET state='scope_changed' WHERE mission_id=? AND state='queued'",(mid,))
            self.store.event(mid,'scope',f'Périmètre version {scope["version"]} enregistré. Les actions futures sont recontrôlées.'); return scope
    def detail(self,mid):
        return {'mission':self.get(mid),'events':self.store.query('SELECT * FROM events WHERE mission_id=? ORDER BY id',(mid,)),
                'evidence':self.store.query('SELECT * FROM evidence WHERE mission_id=? ORDER BY created',(mid,)),
                'actions':self.store.query('SELECT id,tool,state,scope_version,created FROM actions WHERE mission_id=? ORDER BY created',(mid,)),
                'agents':self.store.query('SELECT * FROM agents WHERE mission_id=?',(mid,)),
                'edges':self.store.query('SELECT * FROM edges WHERE mission_id=?',(mid,))}
    def check(self,mid):
        if self.cancels.get(mid,threading.Event()).is_set() or self.get(mid)['stop']:
            raise Fault('CANCELLED','La mission a été interrompue.')
    def action(self,mid,tool,args,action_id=None):
        """Every tool enters here; no free-form shell, tool code or model-issued scope changes."""
        allowed={'code.list','code.read','code.search','static.python','http.get','lab.verify','terminal.local'}
        if tool not in allowed: raise Fault('TOOL_DENIED','Outil indisponible ou non autorisé.')
        aid=action_id or uid()
        fingerprint=hashlib.sha256(json.dumps(args,sort_keys=True).encode()).hexdigest()
        with self.guard:
            self.check(mid); m=self.get(mid)
            old=self.store.query('SELECT * FROM actions WHERE id=?',(aid,))
            if old:
                row=old[0]
                prior=json.loads(row['args'])
                prior_fingerprint=prior.get('fingerprint') if isinstance(prior,dict) else None
                if row['mission_id']!=mid or row['tool']!=tool or prior_fingerprint!=fingerprint: raise Fault('ACTION_COLLISION','Identifiant d’action déjà utilisé pour une autre opération.')
                if row['state']=='done': return json.loads(row['result'])
                raise Fault('ACTION_UNCERTAIN','Action interrompue ou déjà engagée. Examiner sa preuve avant une nouvelle exécution.')
            scope=Scope(m['scope'])
            if tool.startswith('code.') or tool in ['static.python','lab.verify','terminal.local']: scope.file(args.get('path',''))
            if tool=='http.get': scope.url(args['url'])
            recorded=json.dumps({'fingerprint':fingerprint,'parameters':json.loads(redact(json.dumps(args,sort_keys=True)))})
            self.store.execute('INSERT INTO actions VALUES(?,?,?,?,?,?,?,?)',(aid,mid,tool,recorded,'running',m['scope']['version'],None,now()))
            self.active_tools.setdefault(mid,set()).add(aid)
        try:
            result=self.perform(mid,tool,args)
            self.store.execute("UPDATE actions SET state='done',result=? WHERE id=?",(json.dumps(result),aid))
            return result
        except Exception as e:
            self.store.execute("UPDATE actions SET state='failed',result=? WHERE id=?",(json.dumps({'error':redact(e)}),aid))
            self.store.event(mid,'tool_error',f'{tool} : {redact(e)}'); raise
        finally:
            with self.guard: self.active_tools.get(mid,set()).discard(aid)
    def files(self,mid,path):
        scope=Scope(self.get(mid)['scope']); root=scope.file(path)
        if root.is_file(): return [root]
        found=[]
        for base,dirs,names in os.walk(root,followlinks=False):
            dirs[:]=[x for x in dirs if x not in ['.git','node_modules','.venv','venv','__pycache__','build','dist'] and not x.startswith('.')]
            for name in sorted(names):
                self.check(mid)
                p=pathlib.Path(base)/name
                if p.is_symlink(): continue
                try:
                    with self.guard: p=Scope(self.get(mid)['scope']).file(p)
                    info=p.stat()
                    if stat.S_ISREG(info.st_mode) and info.st_size<=512_000: found.append(p)
                except (Fault,OSError): continue
                if len(found)>=5000: return found
        return found
    def read(self,mid,path):
        # Keep the scope version fixed until the bounded read is complete. The
        # opened descriptor is checked, including parent directory symlink races.
        with self.guard:
            self.check(mid); scope=Scope(self.get(mid)['scope']); p=scope.file(path)
            before=p.stat()
            if not stat.S_ISREG(before.st_mode): raise Fault('NOT_REGULAR_FILE','Seuls les fichiers réguliers peuvent être lus.')
            if before.st_size>512_000: raise Fault('FILE_TOO_LARGE','Lecture limitée à 512 Ko par fichier.')
            fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            try:
                opened=os.fstat(fd)
                if not stat.S_ISREG(opened.st_mode): raise Fault('NOT_REGULAR_FILE','Le fichier ouvert n’est pas un fichier régulier.')
                if (opened.st_dev,opened.st_ino)!=(before.st_dev,before.st_ino): raise Fault('SOURCE_CHANGED','La source a changé pendant son ouverture. Relancer la lecture.')
                if sys.platform=='darwin':
                    import fcntl
                    actual=fcntl.fcntl(fd,50,b'\0'*1024).split(b'\0',1)[0].decode('utf-8')
                    scope.file(actual)
                elif pathlib.Path('/proc/self/fd').exists():
                    scope.file(os.readlink(f'/proc/self/fd/{fd}'))
                else: raise Fault('UNSAFE_FILE','Ce système ne permet pas de vérifier le chemin réellement ouvert.')
                data=os.read(fd,512_001)
                after=os.fstat(fd)
                if (opened.st_size,opened.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or len(data)>512_000:
                    raise Fault('SOURCE_CHANGED','La source a changé pendant sa lecture. Relancer la lecture.')
            finally: os.close(fd)
        if b'\0' in data: raise Fault('BINARY_FILE','Fichier binaire : lecture textuelle ignorée.')
        return data.decode('utf-8',errors='replace'),hashlib.sha256(data).hexdigest()
    def perform(self,mid,tool,args):
        if tool=='code.list':
            paths=self.files(mid,args['path']); body='\n'.join(str(p) for p in paths)
            eid=self.store.evidence(mid,'code_map','Cartographie des sources',body,args['path'])
            return {'files':[str(p) for p in paths],'evidence_id':eid,'limit':5000,'truncated':len(paths)>=5000}
        if tool=='code.read':
            body,digest=self.read(mid,args['path']); eid=self.store.evidence(mid,'source',pathlib.Path(args['path']).name,body,args['path'])
            return {'text':redact(body),'source_sha256':digest,'evidence_id':eid}
        if tool=='code.search':
            needle=str(args.get('query',''))
            if not needle or len(needle)>200: raise Fault('INVALID_QUERY','Recherche littérale de 1 à 200 caractères attendue.')
            matches=[]
            for p in self.files(mid,args['path']):
                try: body,digest=self.read(mid,p)
                except (Fault,OSError): continue
                for n,line in enumerate(body.splitlines(),1):
                    if needle.casefold() in line.casefold(): matches.append({'path':str(p),'line':n,'text':redact(line[:400]),'sha256':digest})
                    if len(matches)>=200: break
                if len(matches)>=200: break
            eid=self.store.evidence(mid,'search',f'Recherche : {needle}',json.dumps(matches,ensure_ascii=False),args['path'])
            return {'matches':matches,'evidence_id':eid,'truncated':len(matches)>=200}
        if tool=='static.python':
            matches=[]; parsed=0; errors=[]
            for p in self.files(mid,args['path']):
                if p.suffix!='.py': continue
                self.check(mid)
                try: text,digest=self.read(mid,p); tree=ast.parse(text); parsed+=1
                except (SyntaxError,Fault,OSError) as e: errors.append({'path':str(p),'error':str(e)}); continue
                for node in ast.walk(tree):
                    if not isinstance(node,ast.Call): continue
                    name=ast.unparse(node.func)
                    rule=None
                    if name in ['eval','exec']: rule='Exécution dynamique à examiner'
                    if name in ['pickle.loads','pickle.load']: rule='Désérialisation pickle à examiner'
                    if name in ['subprocess.run','subprocess.call','subprocess.Popen','subprocess.check_output'] and any(k.arg=='shell' and isinstance(k.value,ast.Constant) and k.value.value is True for k in node.keywords): rule='Commande avec shell=True à examiner'
                    if rule:
                        detail={'path':str(p),'line':node.lineno,'rule':rule,'expression':ast.get_source_segment(text,node),'source_sha256':digest,'status':'hypothesis'}
                        eid=self.store.evidence(mid,'hypothesis',rule,json.dumps(detail,ensure_ascii=False),f'{p}:{node.lineno}','hypothesis'); detail['evidence_id']=eid; matches.append(detail)
            summary={'parsed_files':parsed,'hypotheses':matches,'errors':errors,'method':'AST Python ; règles limitées ; aucun résultat ne constitue une vulnérabilité confirmée.'}
            self.store.evidence(mid,'static_analysis','Analyse statique Python',json.dumps(summary,ensure_ascii=False),args['path'])
            return summary
        if tool=='http.get': return self.http_get(mid,args['url'])
        if tool=='terminal.local': return self.terminal(mid,args)
        if tool=='lab.verify': return self.lab(mid,args['path'])
    def _network_check(self,mid,url,version,deadline):
        self.check(mid)
        if time.monotonic()>=deadline: raise Fault('HTTP_DEADLINE','La requête HTTP a atteint sa limite totale de 8 secondes.')
        m=self.get(mid)
        if m['scope']['version']!=version: raise Fault('SCOPE_CHANGED','La requête a été annulée après modification du périmètre.')
        Scope(m['scope']).url(url)
    def _resolve_dns(self,mid,hostname,port,url,version,deadline):
        # OS resolution runs in an owned short-lived child. A stalled resolver
        # cannot occupy an immortal thread or block Pause/Stop. No shell/code
        # supplied by the model is executed by this helper.
        command=[sys.executable]+([] if getattr(sys,'frozen',False) else [str(pathlib.Path(__file__).resolve())])+['--resolve',hostname,str(port)]
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},start_new_session=True)
        try:
            while proc.poll() is None:
                self._network_check(mid,url,version,deadline)
                time.sleep(.03)
            output,errors=proc.communicate(timeout=.5)
            if proc.returncode or len(output)>64_000: raise Fault('DNS_FAILED','La résolution DNS a échoué.',redact(errors.decode(errors='replace')[:200]))
            ips=json.loads(output)
            if not isinstance(ips,list) or not ips: raise Fault('DNS_FAILED','Aucune adresse réseau trouvée.')
            return set(ips)
        finally:
            if proc.poll() is None:
                import signal
                os.killpg(proc.pid,signal.SIGKILL)
            proc.communicate()
    def http_get(self,mid,url):
        """GET only, checked IP pin, no redirect/proxy/cookie, hard total deadline."""
        deadline=time.monotonic()+HTTP_DEADLINE_SECONDS
        with self.guard:
            self.check(mid); m=self.get(mid); u,origin=Scope(m['scope']).url(url); version=m['scope']['version']
        port=u.port or (443 if u.scheme=='https' else 80)
        literal=False
        try: ips={str(ipaddress.ip_address(u.hostname))}; literal=True
        except ValueError: ips=self._resolve_dns(mid,u.hostname,port,url,version,deadline)
        if any(not ipaddress.ip_address(i).is_global for i in ips) and not (literal and all(ipaddress.ip_address(i).is_loopback for i in ips)):
            raise Fault('PRIVATE_ADDRESS','Les adresses privées et changements DNS vers le réseau local sont refusés.','Le laboratoire doit utiliser une origine explicite 127.0.0.1.')
        holder=[None]; finished=threading.Event(); failure=[]
        def watch():
            while not finished.wait(.03):
                try: self._network_check(mid,url,version,deadline)
                except Fault as fault:
                    failure.append(fault)
                    sock=holder[0]
                    if sock is not None:
                        with contextlib.suppress(OSError): sock.shutdown(socket.SHUT_RDWR)
                        with contextlib.suppress(OSError): sock.close()
                    return
        watcher=threading.Thread(target=watch,daemon=True,name='ronin-http-guard'); watcher.start()
        sock=None
        try:
            self._network_check(mid,url,version,deadline)
            ip=sorted(ips)[0]
            sock=socket.create_connection((ip,port),timeout=max(.05,min(2,deadline-time.monotonic()))); holder[0]=sock
            if u.scheme=='https':
                sock=ssl.create_default_context().wrap_socket(sock,server_hostname=u.hostname,do_handshake_on_connect=False); holder[0]=sock
                sock.do_handshake()
            path=urllib.parse.urlunsplit(('','',u.path or '/',u.query,''))
            if any(c in path+u.hostname for c in ['\r','\n']): raise Fault('INVALID_URL','Caractère interdit dans l’URL.')
            # Scope restrictions are serialized with the actual request send.
            with self.guard:
                self._network_check(mid,url,version,deadline)
                sock.sendall(f'GET {path} HTTP/1.1\r\nHost: {u.netloc}\r\nUser-Agent: RONIN-Research/2\r\nConnection: close\r\n\r\n'.encode('ascii'))
            import http.client
            response=http.client.HTTPResponse(sock); response.begin(); chunks=[]; length=0
            while length<256_001:
                self._network_check(mid,url,version,deadline)
                chunk=response.read1(min(32_768,256_001-length))
                if not chunk: break
                chunks.append(chunk); length+=len(chunk)
            self._network_check(mid,url,version,deadline)
            if failure: raise failure[0]
            data=b''.join(chunks); body=redact(data[:256_000].decode('utf-8',errors='replace'))
            result={'url':redact(url),'ip':ip,'status':response.status,'body':body,'truncated':len(data)>256_000,'redirect_followed':False}
            result['evidence_id']=self.store.evidence(mid,'http',f'HTTP {response.status}',json.dumps(result,ensure_ascii=False),url)
            return result
        except (OSError,ValueError) as error:
            if failure: raise failure[0]
            self._network_check(mid,url,version,deadline)
            raise Fault('HTTP_FAILED','La requête HTTP a échoué.',redact(error)) from error
        finally:
            finished.set()
            if sock is not None:
                with contextlib.suppress(OSError): sock.close()
            watcher.join(timeout=.2)
    def terminal(self,mid,args):
        # Only a sandboxed local process; never an unconfined fallback.
        if sys.platform!='darwin' or not pathlib.Path('/usr/bin/sandbox-exec').exists(): raise Fault('SANDBOX_UNAVAILABLE','Le terminal isolé est indisponible sur ce système.')
        mission=self.get(mid); version=mission['scope']['version']; path=Scope(mission['scope']).file(args['path'])
        argv=args.get('argv',[])
        if not isinstance(argv,list) or not argv or not all(isinstance(x,str) for x in argv): raise Fault('INVALID_COMMAND','Liste d’arguments attendue.')
        # Commands see only redacted copies of explicitly authorized source files.
        flags={'/usr/bin/file':{'--brief','--mime-type'},'/usr/bin/wc':{'-l','-w','-c'},'/usr/bin/diff':{'-u','-q'}}
        if argv[0] not in flags: raise Fault('COMMAND_DENIED','Commande non disponible dans le terminal isolé initial.')
        workspace=self.store.home/'workspaces'/mid/uid(); workspace.mkdir(parents=True,exist_ok=True,mode=0o700)
        mapped=[argv[0]]; count=0
        for arg in argv[1:]:
            if arg in flags[argv[0]]: mapped.append(arg); continue
            if arg.startswith('-'): raise Fault('ARGUMENT_DENIED','Option non disponible pour cet outil local.')
            source=Scope(self.get(mid)['scope']).file(path/arg if not pathlib.Path(arg).is_absolute() else arg)
            text,_=self.read(mid,source); copy=workspace/f'source-{count}{source.suffix}'
            copy.write_text(redact(text)); os.chmod(copy,0o600); mapped.append(str(copy)); count+=1
        if count==0 or count>20 or (argv[0]=='/usr/bin/diff' and count!=2): raise Fault('ARGUMENT_DENIED','Choisir de 1 à 20 fichiers ; exactement deux pour diff.')
        esc=lambda v: json.dumps(str(v))
        profile='(version 1)(deny default)(allow process*)(allow sysctl-read)(allow mach-lookup)(allow file-read* (subpath "/System") (subpath "/usr") (subpath "/bin") (subpath "/Library") (literal "/") (literal "/dev/null"))(allow file-read* file-write* (subpath '+esc(workspace)+'))(deny network*)'
        output=workspace/'output.txt'
        capture=output.open('wb')
        proc=subprocess.Popen(['/usr/bin/sandbox-exec','-p',profile]+mapped,cwd=workspace,env={'PATH':'/usr/bin:/bin','HOME':str(workspace),'TMPDIR':str(workspace)},stdout=capture,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+15
        try:
            while proc.poll() is None:
                self.check(mid)
                if self.get(mid)['scope']['version']!=version: raise Fault('SCOPE_CHANGED','Le terminal a été arrêté après modification du périmètre.')
                if time.monotonic()>deadline: raise Fault('TOOL_TIMEOUT','Le terminal a atteint sa limite de 15 secondes.')
                if output.stat().st_size>256_000: raise Fault('TOOL_OUTPUT_LIMIT','La sortie du terminal a atteint 256 Ko.')
                time.sleep(.05)
            capture.flush(); data=output.read_bytes()[:256_001]
        finally:
            if proc.poll() is None:
                import signal
                os.killpg(proc.pid,signal.SIGKILL); proc.wait()
            capture.close()
        result={'argv':argv,'exit_code':proc.returncode,'output':redact(data[:256_000].decode(errors='replace')),'truncated':len(data)>256_000,'isolation':'sandbox-exec, deny network, redacted file copies only, bounded writes'}
        result['evidence_id']=self.store.evidence(mid,'terminal','Commande locale isolée',json.dumps(result,ensure_ascii=False),str(path)); return result
    def lab(self,mid,path):
        """Verify a narrow AST claim with an independent bounded toy reproduction.
        No target code execution; a laboratory result cannot confirm a target finding.
        """
        path=Scope(self.get(mid)['scope']).file(path)
        files=self.files(mid,path); tests=[]
        for p in files:
            if p.suffix!='.py': continue
            text,digest=self.read(mid,p)
            try: tree=ast.parse(text)
            except SyntaxError: continue
            direct=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='eval' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='user_expression' for n in ast.walk(tree))
            # Independent observable reproduction of the exact expression in a controlled built-in lab.
            if direct:
                value=eval('6 * 7',{'__builtins__':{}},{})
                tests.append({'path':str(p),'source_sha256':digest,'claim':'appel direct eval(user_expression) présent','positive_input':'6 * 7','positive_observed':value,'negative_input':'inconnu','negative_observed':'NameError','status':'reproduced_lab_only'})
                try: eval('inconnu',{'__builtins__':{}},{})
                except NameError: pass
                else: raise Fault('COUNTERTEST_FAILED','Le contre-test de laboratoire a échoué.')
        result={'tests':tests,'positive_count':len(tests),'scope':'Reproduction de la primitive dans un laboratoire intégré. Ne prouve ni joignabilité ni impact dans le dépôt.','target_vulnerability_confirmed':False}
        result['evidence_id']=self.store.evidence(mid,'lab','Reproduction contrôlée et contre-test',json.dumps(result,ensure_ascii=False),str(path),'reproduced_lab_only' if tests else 'not_reproduced')
        return result
    def start(self,mid):
        with self.guard:
            self.get(mid)
            if mid in self.workers and self.workers[mid].is_alive(): return {'already_running':True}
            if self.active_tools.get(mid): raise Fault('BUSY','Attendre l’arrêt des outils précédents avant de reprendre.')
            # Only one mission consumes tools at a time in this initial measured runtime.
            if any(t.is_alive() for k,t in self.workers.items() if k!=mid): raise Fault('BUSY','Une mission est déjà en analyse. Mettre cette mission en pause avant de poursuivre.')
            self.cancels[mid]=threading.Event(); self.store.execute('UPDATE missions SET stop=0 WHERE id=?',(mid,)); self.state(mid,'preparing')
            t=threading.Thread(target=self.run,args=(mid,),daemon=True); self.workers[mid]=t; t.start(); return {'started':True}
    def run(self,mid):
        try:
            m=self.get(mid); self.state(mid,'analyzing')
            if shutil.disk_usage(self.store.home).free < 512_000_000:
                self.state(mid,'resource_paused'); self.store.event(mid,'resource','Moins de 512 Mo libres dans le volume des données. Libérer de l’espace puis reprendre.'); return
            self.store.execute("UPDATE agents SET state='busy',mission_id=?,task='Préparation locale' WHERE id IN (1,2,3,7,9)",(mid,))
            self.store.event(mid,'decision','Préparation locale déterministe. Une seule inférence lourde est autorisée ; aucun modèle de remplacement.')
            for root in m['scope']['roots']:
                self.check(mid)
                signature=hashlib.sha256()
                for source in self.files(mid,root):
                    self.check(mid)
                    try:
                        _,digest=self.read(mid,source); signature.update((str(source)+digest).encode())
                    except (Fault,OSError) as e: signature.update((str(source)+str(e)).encode())
                version=signature.hexdigest(); kind='prepared:'+hashlib.sha256(root.encode()).hexdigest()
                if self.store.query('SELECT id FROM memory WHERE mission_id=? AND kind=? AND source_version=?',(mid,kind,version)):
                    self.store.event(mid,'decision','Sources inchangées : réutilisation de la préparation locale vérifiée pour '+root); continue
                self.action(mid,'code.list',{'path':root})
                self.action(mid,'static.python',{'path':root})
                self.store.execute('INSERT INTO memory VALUES(?,?,?,?,?,?)',(uid(),mid,kind,redact(root),version,now()))
            self.check(mid)
            self.store.execute('INSERT INTO memory VALUES(?,?,?,?,?,?)',(uid(),mid,'mission','Préparation locale enregistrée ; conclusions limitées aux empreintes de sources capturées.',str(m['scope']['version']),now()))
            self.state(mid,'waiting_model')
            self.generate(mid,max_new_tokens=self.setting('max_new_tokens',32))
            self.check(mid)
            self.state(mid,'completed')
        except Fault as e:
            if e.code!='CANCELLED': self.state(mid,'error'); self.store.event(mid,'error',str(e))
        except Exception as e:
            self.state(mid,'error'); self.store.event(mid,'error',redact(e))
        finally:
            self.store.execute("UPDATE agents SET state='available',mission_id=NULL,task=NULL WHERE mission_id=?",(mid,))
    def generate(self,mid,prompt=None,max_new_tokens=32,reasoning_effort=None):
        """Run the selected GLM through RONIN; generated text cannot execute tools."""
        m=self.get(mid); self.check(mid)
        if self.engine is None: raise Fault('ENGINE_UNAVAILABLE','L’adaptateur natif GLM est absent.')
        selected=self.setting('model_path',os.environ.get('RONIN_GLM_MODEL',''))
        if not selected: raise Fault('MODEL_NOT_SELECTED','Sélectionner le dossier GLM local dans les réglages.')
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=163840: raise Fault('INVALID_TOKEN_BUDGET','Budget entier de 1 à 163840 tokens attendu.')
        if reasoning_effort is not None and reasoning_effort not in ('low','high','max'): raise Fault('INVALID_REASONING_EFFORT','Effort explicite attendu : low, high ou max.')
        if prompt is not None and (not isinstance(prompt,str) or not prompt.strip()): raise Fault('INVALID_PROMPT','Une demande textuelle non vide est nécessaire.')
        cancel=self.cancels.setdefault(mid,threading.Event())
        started=time.monotonic(); text=[]; events=[]; tokens=[]
        session=self.engine.create_session(mid)
        try:
            self.engine.load(selected,cancel=cancel)
            self.check(mid)
            self.store.event(mid,'model_started','Génération GLM-5.3-Flash Q2_K_S par le moteur natif partagé. La sortie du modèle reste une proposition non vérifiée.')
            for event in self.engine.stream([{'role':'user','content':prompt if prompt is not None else m['prompt']}],max_new_tokens=max_new_tokens,cancel=cancel,session_id=session['id'],reasoning_effort=reasoning_effort):
                self.check(mid)
                content=event.get('content','')
                if isinstance(content,str): text.append(content)
                ids=event.get('tokens',[])
                if isinstance(ids,list): tokens.extend(ids)
                if event.get('stop') or 'timings' in event: events.append(event)
            self.check(mid)
            result={'text':''.join(text),'tokens':tokens,'events':events,
                    'seconds':time.monotonic()-started,'model':self.engine.status()['model'],
                    'reasoning_effort':reasoning_effort,'reasoning_effort_scope':'chat_template_conditioning_only',
                    'session_id':session['id'],'verification':'model_output_unverified',
                    'tools_executed_from_output':False}
            result['evidence_id']=self.store.evidence(mid,'model_output','Réponse GLM locale',json.dumps(result,ensure_ascii=False),selected,'model_output_unverified')
            self.store.event(mid,'model_completed',json.dumps({'evidence_id':result['evidence_id'],'seconds':result['seconds'],'tokens':len(tokens)},ensure_ascii=False))
            return result
        except Exception as error:
            if cancel.is_set(): raise Fault('CANCELLED','Génération interrompue.') from error
            raise Fault(getattr(error,'code','MODEL_ERROR'),str(error)) from error
        finally:
            self.engine.close_session(session['id'],mid)
    def stop(self,mid,pause=False):
        with self.guard:
            self.get(mid); self.cancels.setdefault(mid,threading.Event()).set()
            self.store.execute('UPDATE missions SET stop=1,status=?,updated=? WHERE id=?',('paused' if pause else 'stopped',now(),mid))
            self.store.event(mid,'pause' if pause else 'stop','Mise en pause demandée.' if pause else 'Arrêt demandé. Aucun redémarrage automatique.')
        deadline=time.monotonic()+6; t=self.workers.get(mid)
        if t: t.join(timeout=max(0,deadline-time.monotonic()))
        while self.active_tools.get(mid) and time.monotonic()<deadline: time.sleep(.03)
        return {'stopped':not(t and t.is_alive()) and not bool(self.active_tools.get(mid)),'status':self.get(mid)['status']}
    def skills(self,query=''):
        from skills import Catalog
        catalog=Catalog(ROOT/'catalog')
        names=[x['name'] for x in catalog.search(query,limit=50)] if query.strip() else list(catalog.entries)[:50]
        result=[]
        for name in names:
            item=dict(catalog.entries[name]); state=catalog.status(name)
            item['status']=state['label']+' ; scripts '+{'disabled':'désactivés','missing_dependency':'avec dépendances manquantes','incompatible':'incompatibles','not_applicable':'absents'}.get(state['script_state'],state['script_state'])
            item['dependency_status']=state; result.append(item)
        return result
    def skill_read(self,skill_id):
        from skills import Catalog, CatalogError
        try:
            item=Catalog(ROOT/'catalog').read(skill_id)
            return {'metadata':{k:v for k,v in item.items() if k!='content'},'text':item['content'],'trusted':False,'grants_permissions':False}
        except CatalogError as e: raise Fault('SKILL_INVALID',str(e))
    def skill_attach(self,mid,skill_id):
        self.get(mid); document=self.skill_read(skill_id); metadata=document['metadata']
        eid=self.store.evidence(mid,'skill_reference','Méthode chargée : '+skill_id,document['text'],str(metadata.get('source','')),'untrusted_reference')
        self.store.event(mid,'decision','Skill chargée à la demande : '+skill_id+'. Ce document n’accorde aucune permission ; les scripts restent désactivés.')
        return {'evidence_id':eid,'grants_permissions':False,'scope_unchanged':True}
    def export(self,mid,destination=None):
        from report import export_report
        d=self.detail(mid); dest=pathlib.Path(destination) if destination else self.store.home/'Exports'/mid
        return export_report(d,dest,self.status())
    def dispatch(self,method,p):
        mid=p.get('id')
        if method=='status': return self.status()
        if method=='list': return self.store.query('SELECT * FROM missions ORDER BY updated DESC')
        if method=='create': return self.create(p)
        if method=='detail': return self.detail(mid)
        if method=='start': return self.start(mid)
        if method=='pause': return self.stop(mid,True)
        if method=='stop': return self.stop(mid)
        if method=='scope': return self.scope_update(mid,p.get('roots',[]),p.get('origins',[]),p.get('authorized',False))
        if method=='tool': return self.action(mid,p['tool'],p.get('args',{}),p.get('action_id'))
        if method=='export': return self.export(mid,p.get('destination'))
        if method=='comment':
            self.get(mid); self.store.event(mid,'user',p['text']); return {'saved':True}
        if method=='settings':
            if 'model_path' in p:
                path=str(pathlib.Path(p['model_path']).expanduser()) if p['model_path'] else ''
                self.store.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',('model_path',json.dumps(path)))
            return {'model_path':self.setting('model_path')}
        if method=='skills': return self.skills(p.get('query',''))
        if method=='skill_read': return self.skill_read(p['skill_id'])
        if method=='skill_attach': return self.skill_attach(mid,p['skill_id'])
        if method=='memory': return self.store.query('SELECT * FROM memory WHERE mission_id=? ORDER BY created',(mid,))
        if method=='agents': return self.store.query('SELECT * FROM agents ORDER BY id')
        if method=='identity': return {'prompt':IDENTITY,'provenance':'Z.ai · zai-org/GLM-5.3-Flash · AirLLM Next'}
        if method=='generate':
            if set(p)-{'id','prompt','max_new_tokens','reasoning_effort'}: raise Fault('INVALID_ARGUMENT','Option de génération inconnue.')
            return self.generate(mid,p.get('prompt'),p.get('max_new_tokens',32),p.get('reasoning_effort'))
        if method=='engine_load':
            if not self.engine: raise Fault('ENGINE_UNAVAILABLE','Adaptateur GLM absent.')
            return self.engine.load(self.setting('model_path',os.environ.get('RONIN_GLM_MODEL','')))
        if method=='engine_unload': return self.engine.unload() if self.engine else {'loaded':False}
        if method=='shutdown':
            if self.engine: self.engine.request_shutdown()
            for m in self.store.query("SELECT id FROM missions WHERE status NOT IN ('stopped','completed','error','archived')"): self.stop(m['id'],True)
            if self.engine: self.engine.close()
            return {'closed':True}
        raise Fault('UNKNOWN_METHOD','Action inconnue.')

def main():
    os.umask(0o077)
    runtime=Runtime(os.environ.get('RONIN_DATA_HOME'))
    # Responses may arrive out of order and always retain their request id.
    # Control requests use the input thread so four busy tools cannot starve Stop.
    output_lock=threading.Lock(); closing=threading.Event(); slots=threading.BoundedSemaphore(16)
    executor=ThreadPoolExecutor(max_workers=4,thread_name_prefix='ronin-request')
    previous_signals={}
    def terminate(signum, frame):
        if closing.is_set(): return
        closing.set()
        if runtime.engine: runtime.engine.request_shutdown()
        raise SystemExit(128+signum)
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM,signal.SIGINT):
            previous_signals[signum]=signal.getsignal(signum)
            signal.signal(signum,terminate)
    def emit(response):
        with output_lock: print(json.dumps(response,ensure_ascii=False),flush=True)
    def handle(request):
        try:
            if closing.is_set() and request['method']!='shutdown': raise Fault('CANCELLED','L’application se ferme ; cette action ne sera pas lancée.')
            result=runtime.dispatch(request['method'],request.get('params',{}))
            response={'id':request.get('id'),'result':result}
        except Fault as e: response={'id':request.get('id'),'error':e.as_dict()}
        except Exception as e: response={'id':request.get('id'),'error':{'code':'INTERNAL','message':redact(e),'action':'Les preuves déjà enregistrées sont conservées.'}}
        emit(response)
    def pooled(request):
        try: handle(request)
        finally: slots.release()
    try:
        while True:
            line=sys.stdin.readline(1_000_001)
            if not line: break
            request={}
            try:
                if len(line)>1_000_000:
                    while line and not line.endswith('\n'): line=sys.stdin.readline(65_536)
                    raise Fault('REQUEST_TOO_LARGE','Requête trop volumineuse.')
                request=json.loads(line)
                if not isinstance(request,dict) or not isinstance(request.get('method'),str) or not isinstance(request.get('params',{}),dict):
                    request={}; raise Fault('INVALID_REQUEST','Requête structurée attendue.')
                method=request['method']
                if method in {'stop','pause','scope','shutdown'}:
                    if method=='shutdown': closing.set()
                    handle(request)
                    if method=='shutdown': break
                elif slots.acquire(blocking=False): executor.submit(pooled,request)
                else: raise Fault('BUSY','Trop d’actions en attente ; réessayer après leur arrêt.')
            except Fault as error: emit({'id':request.get('id'),'error':error.as_dict()})
            except Exception as error: emit({'id':request.get('id') if isinstance(request,dict) else None,'error':{'code':'INVALID_REQUEST','message':redact(error)}})
    finally:
        closing.set()
        runtime.dispatch('shutdown',{})
        executor.shutdown(wait=True,cancel_futures=False)
        runtime.store.db.close()
        for signum,handler in previous_signals.items(): signal.signal(signum,handler)

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--resolve':
        try:
            addresses=sorted({entry[4][0] for entry in socket.getaddrinfo(sys.argv[2],int(sys.argv[3]),type=socket.SOCK_STREAM)})[:128]
            print(json.dumps(addresses),flush=True)
        except Exception as error:
            print(redact(error),file=sys.stderr); sys.exit(1)
    else: main()
