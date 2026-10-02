"""Convert Round 1 grid export to the map format accepted by main2nos."""
import argparse
import json
import math
from pathlib import Path


def convert_data(data, target_location='standing', include_shot=False):
    size = data['grid_size']
    w,h = int(size['w']),int(size['h'])
    if not 1 <= w <= 50 or not 1 <= h <= 50:
        raise ValueError('Grid dimensions must be 1..50')
    def cell(value):
        x,y = (value['x'],value['y']) if isinstance(value,dict) else value
        if isinstance(x,bool) or isinstance(y,bool) or int(x)!=x or int(y)!=y:
            raise ValueError('Grid coordinates must be integers')
        if not 0 <= x < w or not 0 <= y < h:
            raise ValueError('Grid coordinate outside map: %s' % (value,))
        return [int(x),int(y)]
    targets=[]
    for target in data.get('targets_found',[]):
        if not target.get('is_main_target',True):
            continue
        if not include_shot and (target.get('is_knocked_down') or target.get('ir_status') in ('sent','pending','uncertain')):
            continue
        field='standing_grid' if target_location=='standing' else 'target_grid'
        if field not in target:
            raise ValueError('Target lacks '+field)
        destination=cell(target[field])
        if destination not in targets:targets.append(destination)
    walls=[]
    for name,dx,dy in (('N_wall_matrix',0,1),('E_wall_matrix',1,0)):
        matrix=data[name]
        if len(matrix)!=h or any(len(row)!=w for row in matrix):
            raise ValueError(name+' must have grid height rows and grid width columns')
        for y in range(h):
            for x in range(w):
                # Outer boundaries are implicit in main2nos.Maze.edge.
                if x+dx < w and y+dy < h and matrix[y][x]:
                    walls.append([[x,y],[x+dx,y+dy]])
    cell_size=float(data.get('cell_size_m',.60))
    if not math.isfinite(cell_size) or not .2 <= cell_size <= 2:
        raise ValueError('Invalid cell size')
    return dict(grid_size={'w':w,'h':h},start=cell(data['start_grid']),
                initial_heading_deg=0,cell_size_m=cell_size,targets=targets,walls=walls)


def convert():
    folder=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=folder/'Map_Data_Round1.json')
    parser.add_argument('--output',type=Path,default=folder/'map_config.json')
    parser.add_argument('--target-location',choices=('standing','target'),default='standing',
                        help='standing: observed shooting cell; target: physical target cell')
    parser.add_argument('--include-shot',action='store_true',help='Include targets whose IR was already sent')
    args=parser.parse_args()
    try:
        data=json.loads(args.input.read_text(encoding='utf-8-sig'))
        converted=convert_data(data,args.target_location,args.include_shot)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(converted,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temporary.replace(args.output)
    print('Saved: %s\nNavigation targets: %d; internal walls: %d; target location: %s' %
          (args.output,len(converted['targets']),len(converted['walls']),args.target_location))
    if len(converted['targets'])>8:
        print('main2nos supports at most 8 navigation targets; select a smaller set before running.')
    if not converted['targets']:
        print('No remaining targets. main2nos will ask for manual navigation coordinates.')
    print('Place the robot at the configured start cell center, facing N (+y), before main2nos.')


if __name__=='__main__':
    convert()
