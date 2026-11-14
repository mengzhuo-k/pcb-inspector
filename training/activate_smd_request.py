"""Validate and install a completed, registered SMD archive classifier."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from defect_classifier import DefectClassifier
import platform_app as platform


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request-dir',type=Path,required=True)
    args=parser.parse_args()
    folder=args.request_dir.resolve()
    state_path=folder/'state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    if state['state'] not in ('ready','installed'):
        raise RuntimeError('training has not completed')
    torch.set_num_threads(8)
    weights=Path(state['weights'])
    card_path=folder/'registered-card.json'
    card=json.loads(card_path.read_text(encoding='utf-8'))
    candidate=DefectClassifier(weights,card_path)
    if not candidate.available():raise RuntimeError(candidate.last_error)
    with Image.open(state['normal_example']) as image:
        normal=candidate.predict(image.convert('RGB'))
    if normal['key']!='good_solder' or normal['is_defect']:
        raise RuntimeError('normal reference failed sanity check: '+json.dumps(normal))
    card.update(model_id=state['model_id'],use_with_sites=True,allow_unaligned=True,authoritative_board_decision=True)
    initial=ROOT/'training/yolo11n-cls.pt'
    card['initial_model_sha256']=hashlib.sha256(initial.read_bytes()).hexdigest()
    target_weights=ROOT/'models/smd_defect_cls_v1.pt'
    target_card=ROOT/'models/smd_defect_cls_v1.json'
    backup=folder/'previous-classifier';backup.mkdir(exist_ok=True)
    for path in (target_weights,target_card):
        if path.is_file() and not (backup/path.name).exists():shutil.copy2(path,backup/path.name)
    temporary_weights=target_weights.with_suffix('.new.pt')
    temporary_card=target_card.with_suffix('.new.json')
    shutil.copy2(weights,temporary_weights)
    temporary_card.write_text(json.dumps(card,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary_weights,target_weights)
    os.replace(temporary_card,target_card)
    with platform.db() as conn:
        conn.execute("INSERT OR REPLACE INTO platform_settings VALUES('active_classifier_model',?)",(state['model_id'],))
        conn.execute('UPDATE models SET published=1,card=? WHERE id=?',(json.dumps(card,ensure_ascii=False),state['model_id']))
        conn.execute("UPDATE model_updates SET state='ready',trained_revision=max(trained_revision,target_revision),message=? WHERE project_id=? AND job_id=?",
            ('SMD DefDB 七类分类模型已完成训练和独立测试',state['project_id'],state['job_id']))
    state.update(state='installed',normal_check=normal)
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'model_version':state['model_version'],'model_id':state['model_id'],
        'normal_example':normal,'metrics':state['metrics']},ensure_ascii=True))


if __name__=='__main__':main()
