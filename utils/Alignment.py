from minineedle import needle, smith, core
import pandas as pd

def alignment(gt_labels, pred_labels):
    """
    Alignment between ground truth labels and predicted labels.
    :param gt_labels: ground truth labels
    :param pred_labels: predicted labels
    :return: alignment score
    """
    alignment = needle.NeedlemanWunsch(gt_labels, pred_labels)
    alignment.align()
    gt, pred = alignment.get_aligned_sequences('list')
    gt = ['-' if isinstance(x, core.Gap) else x for x in gt]
    pred = ['-' if isinstance(x, core.Gap) else x for x in pred]
    return gt, pred

def save_alignment_res(gt_labels, pred_labels, save_path):
    gt, pred = alignment(gt_labels, pred_labels)
    assert len(gt) == len(pred)
    relation, values = analyze_alignment(gt, pred)
    assert len(gt) == len(relation)
    
    dict = {
        '真实标签': gt,
        '关系': relation,
        '预测标签': pred,
    }
    df = pd.DataFrame(dict)
    md_str = df.to_markdown()
    with open(save_path, 'w') as f:
        f.write(md_str)
    return values

def analyze_alignment(gts, preds):
    assert len(gts) == len(preds)
    
    duojian = 0
    loujian = 0
    leixing = 0
    shizhi = 0
    puhao = 0
    diaohao = 0
    paihao = 0
    yuyi_yingao = 0
    putong_yingao = 0
    weizhi_yingao = 0
    qita = 0
    
    relation = []
    for gt, pred in zip(gts, preds):
        if gt == pred:
            relation.append('-------')
            continue
        if gt == '-':
            relation.append('多检')
            duojian += 1
        elif pred == '-':
            relation.append('漏检')
            loujian += 1
        else:
            if gt == '[UNKNOWN]' or pred == '[UNKNOWN]':
                relation.append('类型')
                leixing += 1
            elif gt == 'tie' or pred == 'tie':
                relation.append('类型')
                leixing += 1
            elif gt == 'tercet' or pred == 'tercet':
                relation.append('类型')
                leixing += 1
            elif gt.startswith('rest'):
                if pred.startswith('rest'):
                    relation.append('时值')
                    shizhi += 1
                else:
                    relation.append('类型')
                    leixing += 1
            elif gt.startswith('clef'):
                if pred.startswith('clef'):
                    relation.append('谱号')
                    puhao += 1
                else:
                    relation.append('类型')
                    leixing += 1
            elif gt.startswith('keySignature'):
                if pred.startswith('keySignature'):
                    relation.append('调号')
                    diaohao += 1
                else:
                    relation.append('类型')
                    leixing += 1
            elif gt.startswith('timeSignature'):
                if pred.startswith('timeSignature'):
                    relation.append('拍号')
                    paihao += 1
                else:
                    relation.append('类型')
                    leixing += 1
            elif gt.startswith('note'):
                temp_r = []
                if not pred.startswith('note'):
                    temp_r.append('类型')
                    leixing += 1
                else:
                    gt_pitch = gt.split('_')[0].split('-')[-1]
                    pred_pitch = pred.split('_')[0].split('-')[-1]

                    if gt_pitch != pred_pitch:
                        if len(gt_pitch) == 2 and len(pred_pitch) == 2:
                            temp_r.append('普通音高')
                            putong_yingao += 1
                        elif len(gt_pitch) == 2 and len(pred_pitch) == 3:
                            if gt_pitch[0] == pred_pitch[0] and gt_pitch[1] == pred_pitch[-1]:
                                temp_r.append('语义音高')
                                yuyi_yingao += 1
                            else :
                                temp_r.append('普通音高')
                                putong_yingao += 1
                        elif len(gt_pitch) == 3 and len(pred_pitch) == 2:
                            if gt_pitch[0] == pred_pitch[0] and gt_pitch[-1] == pred_pitch[1]:
                                temp_r.append('语义音高')
                                yuyi_yingao += 1
                            else :
                                temp_r.append('普通音高')
                                putong_yingao += 1
                        elif len(gt_pitch) == 3 and len(pred_pitch) == 3:
                            if gt_pitch[0] == pred_pitch[0] and gt_pitch[-1] == pred_pitch[-1]:
                                temp_r.append('语义音高')
                                yuyi_yingao += 1
                            else :
                                temp_r.append('普通音高')
                                putong_yingao += 1
                        else:
                            temp_r.append('未知音高')
                            weizhi_yingao += 1
                            
                    gt_duration = '_'.join(gt.split('_')[1:])
                    pred_duration = '_'.join(pred.split('_')[1:])
                    if gt_duration != pred_duration:
                        temp_r.append('时值')
                        shizhi += 1
                relation.append('+'.join(temp_r))
            else:
                relation.append('其他')
                qita += 1
                
    
    values = {
        "多检": duojian,
        "漏检": loujian,
        "类型": leixing,
        "时值": shizhi,
        "谱号": puhao,
        "调号": diaohao,
        "拍号": paihao,
        "普通音高": putong_yingao,
        "语义音高": yuyi_yingao,
        "未知音高": weizhi_yingao,
        "其他": qita,
    }
        
    return relation, values
    
