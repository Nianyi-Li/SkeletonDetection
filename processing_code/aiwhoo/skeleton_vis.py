# This module is for visualizing the output skeleton from a SMAP run. 
# it is being used for evaluating the 3D coordinate system for eventually evaluating speed
import json
import pdb

def write_line_obj(verts,lines,filename):
    fp = open(filename,'w')
    for vert in verts:
        fp.write('v ' + str(vert[0]) + ' ' + str(vert[1]) + ' ' + str(vert[2]) + '\n')
    for line in lines:
        fp.write('l')
        for lindex in line:
            fp.write(' ' + str(lindex + 1))
        fp.write('\n')
    fp.close()
if __name__ == '__main__':
    smap_json_file = "/home/scott/Documents/sidehustle/covid/SMAP/model_logs/stage3_root2/result/stage3_root2_run_inference_test_.json"
    out_dir = '/home/scott/Documents/sidehustle/covid/data/skel_vis/'

    with open(smap_json_file, 'r') as f:
        data = json.load(f)['3d_pairs']
    for dat in data:
        file_pref = dat['image_path'].replace('.jpg','')
        dat3d = dat['pred_3d']
        verts = []
        lines = []
        offset = 0
        num_valid_person = 0
        for person in dat3d:
            if len(person) ==15:
                tpts = []
                for v4d in person:
                    #do a validity check on each point
                    # if pt[3] == 1: idk
                    tpts.append([v4d[0],v4d[1],v4d[2]])
                if len(tpts) == 15:
                    verts.extend(tpts)
                    
                    #LINES
                    # head to neck
                    lines.append([1+offset,offset])
                    # neck to Lshoulder
                    lines.append([offset,3+offset])
                    # neck to Rshoulder
                    lines.append([offset,9+offset])
                    # Lshoulder to Lelebow
                    lines.append([3+offset,4+offset])
                    # Rshoulder to Relbow
                    lines.append([9+offset,10+offset])
                    # Lelbow to Lwrist
                    lines.append([4+offset,5+offset])
                    # Relbow to Rwrist
                    lines.append([10+offset,11+offset])
                    # neck to pelvis
                    lines.append([offset,2+offset])
                    # pelvis to Lhip
                    lines.append([2+offset,6+offset])
                    # pelvis to Rhip
                    lines.append([2+offset,12+offset])
                    # Lhip to Lknee
                    lines.append([6+offset,7+offset])
                    # Rhip to Rknee
                    lines.append([12+offset,13+offset])
                    # Lknee to Lankle
                    lines.append([7+offset,8+offset])
                    # Rknee to Rankle
                    lines.append([13+offset,14+offset])
                    offset = offset + 15
        out_file = out_dir + file_pref + '.obj'
        write_line_obj(verts,lines,out_file)
