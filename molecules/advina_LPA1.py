from .conversion import mol_to_smiles

def adock(receptor_input,
        smiles,
        ligand_name,
        center_x=14.444,
        center_y=5.250,
        center_z=-18.278,
        size_x=20,
        size_y=20,
        size_z=20,
 #      vina = 'autodockvina',
        vina='qvina2',
        seed=None,
        cpu=33,
        lig_dir = './lpa1_scores/ligand_files/',
        out_dir = './lpa1_scores/output/',
        log_dir = './lpa1_scores/log/',
        conf_dir = './lpa1_scores/config/'):

    #Imports
    import os
    import subprocess
    import psutil
    import re

    timeout_duration = 1000

    #mkdir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(lig_dir, exist_ok=True)

    ligand = lig_dir + ligand_name + '.pdbqt'
    output = out_dir + ligand_name + '_out.pdbqt'
    config = conf_dir + ligand_name + '.conf'
    log = log_dir + ligand_name + '_log.txt'

    #Convert smiles to PDBQT ligand file
    if not os.path.isfile(ligand):
        with subprocess.Popen('obabel -:"' + smiles + '" -O ' + ligand + ' -h --gen3d' + ' > /dev/null 2>&1', \
                shell=True, start_new_session=True) as proc:
            try:
                proc.wait(timeout=timeout_duration)
            except subprocess.TimeoutExpired:
                p = psutil.Process(proc.pid)
                p.terminate()
    else:
        print(f'Ligand file: {ligand!r} already exists.')

    #Dock
    if os.path.isfile(receptor_input):
        if not os.path.isfile(output):
            
 #Create configuration files
            conf = 'receptor = ' + receptor_input + '\n' +\
                    'ligand = ' + ligand + '\n' + \
                    'center_x = ' + str(center_x) + '\n' + \
                    'center_y = ' + str(center_y) + '\n' + \
                    'center_z = ' + str(center_z) + '\n' + \
                    'size_x = ' + str(size_x) + '\n' + \
                    'size_y = ' + str(size_y) + '\n' + \
                    'size_z = ' + str(size_z) + '\n' + \
                    'out = ' + output + '\n' + \
                    'cpu = ' + str(cpu)
            
            if seed is not None:
                conf += '\n' \
                    'seed = ' + str(seed)

            with open(config, 'w') as f:
                f.write(conf)
                
  #Run the docking simulation      But The Docking simu is NOT Found
            with subprocess.Popen('' + vina + \
                    ' --config ' + config + \
                    ' --log ' + log + \
                    ' > /dev/null 2>&1', \
                    shell=True, start_new_session=True) as proc:
                try:
                    proc.wait(timeout=timeout_duration)
                except subprocess.TimeoutExpired:
                    p = psutil.Process(proc.pid)
                    p.terminate()
        result = 0
  
  # Parse the docking score  
        try:
            score = float("inf")
            with open(output, 'r') as f:
                for line in f.readlines():
                    if "REMARK VINA RESULT" in line:
                        new_score = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', line)[0]
                        score = min(score, float(new_score))
                result = score
                
#errors happens here              
        except FileNotFoundError:
            print('Empty--', ligand_name)
            result = 1001
           
        
    else:
        print(f'Protein file: {receptor_input!r} not found!')
        result = 0

    return (result)
# the simulation failed to produce any output
# the ligand files were generated successfully, but the docking simulation failed for all ligands
# incorrect parameters or an issue with the protein file or AutoDock Vina software

def calculateDockingScore(mol):
    # targeting protein file for docking 
    protein_surface = '/home/student5/Downloads/Thesis_Madiha/old_fragmentation/fragvae-CA9-GPX4-LPA1/DATA/protein_files/LPA1.pdbqt'
    # creating appropriate file names for ligands
    smi = mol_to_smiles(mol)
    ligand_name = smi.replace('(', '{').replace(')', '}').replace('#','-')
    return adock(protein_surface, smi, ligand_name)


#import pandas as pd
#molecules = pd.read_csv('/home/student5/Downloads/fragvae_three_targets/DATA/ZINC/PROCESSED/train.smi')
#molecules = molecules.iloc[:,1]

#docking_scores_LPA1 = []
#for smi in molecules:
    #score = calculateDockingScore(smi)
    #docking_scores_LPA1.append(score)
    
#pd.DataFrame(docking_scores_LPA1).to_csv('LPA1_Docking_scores.csv')
#
    
#calculateDockingScore(mol, targets)
