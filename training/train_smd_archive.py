"""Train an auditable whole-board model from the user-selected SMD DefDB tar."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import platform_app as platform
import prepare_smd_defdb as prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--request-dir',type=Path,required=True)
    args=parser.parse_args()
    folder=args.request_dir.resolve();folder.mkdir(parents=True,exist_ok=True)
    state_file=folder/'state.json'
    state={}
    def update(**values):
        state.update(values)
        temporary=state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(temporary,state_file)
        print(json.dumps(values,ensure_ascii=True),flush=True)
    with platform.db() as conn:
        project=platform.inspection_workspace(conn)
    pid=project['id'];did=uuid.uuid4().hex;jid=uuid.uuid4().hex
    update(state='preparing',project_id=pid,dataset_id=did,job_id=jid,archive=str(args.archive.resolve()))
    extracted=folder/'originals'
    archive_digest=hashlib.sha256()
    with args.archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            archive_digest.update(chunk)
    count=0
    with tarfile.open(args.archive) as tar:
        for member in tar:
            if not member.isfile():continue
            parts=Path(member.name).parts
            if len(parts)!=2 or parts[0] not in prepare.CLASSES or not parts[1].lower().endswith('.jpg'):
                raise ValueError('unexpected archive entry: '+member.name)
            target=extracted.joinpath(*parts)
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('wb') as output:
                shutil.copyfileobj(source,output)
            count+=1
    if count!=1190:raise ValueError('expected 1190 archive images, got '+str(count))
    prepare.STORE=extracted
    data=prepare.build();assignment=prepare.assign_groups(data['files'])
    cls=folder/'cls';image_folder=platform.IMAGES/pid/('smd-'+did)
    image_folder.mkdir(parents=True,exist_ok=True)
    def convert(item):
        split=assignment[item['group']]
        filename=item['path'].name
        target=cls/split/item['class_name']/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(item['path']) as original:
            picture=ImageOps.exif_transpose(original).convert('RGB')
            picture.thumbnail((512,512))
            picture.save(target,'JPEG',quality=95)
            size=picture.size
        iid=uuid.uuid4().hex
        snapshot_path=image_folder/(iid+'.jpg')
        try:os.link(target,snapshot_path)
        except OSError:shutil.copy2(target,snapshot_path)
        digest=hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        metadata=dict(id=iid,project_id=pid,filename=filename,path=str(snapshot_path),width=size[0],height=size[1],
            batch='SMD DefDB archive',board_serial=item['class_name']+'#'+str(item['group'][1]),camera='dataset.tar',
            primary_class=item['class_name'],sha256=digest,state='labeled',review_status='approved',revision=1,
            alignment=None,created=platform.now(),source_archive=str(args.archive.resolve()))
        return dict(id=iid,split=split,metadata=metadata,source=str(item['path']),group=metadata['board_serial'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(convert,data['files']))
    counts=Counter(row['split'] for row in rows)
    manifest=dict(seed=prepare.SEED,policy='augmentation blocks grouped; byte-identical groups merged',
        archive_sha256=archive_digest.hexdigest(),duplicates=data['duplicates'],counts=dict(counts),
        files=[dict(file=str((cls/row['split']/row['metadata']['primary_class']/row['metadata']['filename']).relative_to(cls)),
                    class_name=row['metadata']['primary_class'],split=row['split'],group=row['group']) for row in rows])
    (cls/'split.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=1),encoding='utf-8')
    summary={**dict(counts), 'class_order':prepare.CLASSES,'classes':dict(Counter(row['metadata']['primary_class'] for row in rows))}
    config=dict(model=str(ROOT/'training/yolo11n-cls.pt'),epochs=40,imgsz=224,batch=32,device='cpu',threads=8,
        classes=prepare.CLASSES,source_archive=str(args.archive.resolve()),archive_sha256=archive_digest.hexdigest(),
        use_with_sites=True,allow_unaligned=True)
    with platform.db() as conn:
        version=conn.execute('SELECT coalesce(max(version),0)+1 FROM datasets WHERE project_id=?',(pid,)).fetchone()[0]
        conn.execute('INSERT INTO datasets(id,project_id,version,template,split_policy,counts,warnings,created_by,created) VALUES(?,?,?,?,?,?,?,?,?)',
            (did,pid,version,'cls','source-photo-group',json.dumps(summary),'[]','User archive training',platform.now()))
        conn.executemany('INSERT INTO dataset_items VALUES(?,?,?)',[(did,row['id'],row['split']) for row in rows])
        conn.executemany('INSERT INTO dataset_snapshots VALUES(?,?,?,?,?)',
            [(did,row['id'],row['split'],json.dumps(row['metadata']), '[]') for row in rows])
        conn.execute('INSERT INTO jobs(id,dataset_id,project_id,template,config,state,created,started,log) VALUES(?,?,?,?,?,?,?,?,?)',
            (jid,did,pid,'cls',json.dumps(config),'running',platform.now(),platform.now(),str(platform.LOGS/(jid+'.log'))))
    run_name='smd_archive_'+did[:12]
    command=[sys.executable,'-u',str(ROOT/'training/train_smd_classifier.py'),'--data',str(cls),
        '--epochs','40','--imgsz','224','--batch','32','--threads','8','--workers','0','--cache','none',
        '--patience','12','--name',run_name,'--version','smd-defdb-'+did[:8],
        '--output-card',str(folder/'classifier.json'),'--output-weights',str(folder/'trained.pt'),
        '--platform-database',str(platform.DB),'--job-id',jid]
    update(state='training',images=len(rows),splits=dict(counts),run_name=run_name,log=str(platform.LOGS/(jid+'.log')),
        normal_example=str(extracted/'good_solder/good_solder_60.jpg'))
    env=os.environ.copy();env.update(PCB_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1')
    with (platform.LOGS/(jid+'.log')).open('w',encoding='utf-8') as log:
        code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError('training process failed: '+str(code))
    card=json.loads((folder/'classifier.json').read_text(encoding='utf-8'))
    with platform.db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        version=platform.next_model_version(conn,pid);mid=uuid.uuid4().hex
        target=platform.MODELS/pid/version/'best.pt';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/'trained.pt',target)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        card.update(name='SMD DefDB 七类全量模型',template='cls',version=version,dataset=did,
                    dataset_info=card['dataset'],source_archive=str(args.archive.resolve()),archive_sha256=archive_digest.hexdigest(),
                    use_with_sites=True,allow_unaligned=True,weights=str(target))
        conn.execute('INSERT INTO models(id,project_id,job_id,version,sha256,metrics,threshold,weights,card,published,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (mid,pid,jid,version,digest,json.dumps(card['metrics']),.5,str(target),json.dumps(card,ensure_ascii=False),0,platform.now()))
        conn.execute("UPDATE jobs SET state='done',model_id=?,weights=?,metrics=?,finished=? WHERE id=?",
            (mid,str(target),json.dumps(card['metrics']),platform.now(),jid))
    (folder/'registered-card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
    update(state='ready',model_id=mid,model_version=version,weights=str(target),metrics=card['metrics'])


if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc()
        raise
