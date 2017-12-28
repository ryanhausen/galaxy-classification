# -*- coding: utf-8 -*-
import os
import pandas as pd

def extract_from_table2(path_to_file, save_new_file=None):
    """
    Gets the fractions for the basic morphology classifications from table_2.dat
    basic classifications are: spheroid, disk, irregular, point source, 
                               unclassifiable

    arguments:
    path_to_file = path to table2.dat
    save_new_file = path to have transformed file save to, if is set to None 
                    then it doesn't save new table to a text file
                    
    returns: pandas dataframe with percentages per each class for each ID
    """    

    # the .dat file prefixes the value of '2epoch' in the depth column with
    # a space so that everything looks even. This however is not helpful when
    # parsing the file, so remove it
  
    tmp_file = path_to_file + '.tmp'
  
    with open(path_to_file, mode='r') as old, open(tmp_file, 'w') as new: 
        for line in old:
            if line[0] == ' ':
                line = line[1:]
            
            new.write(line)
    
    # cols from table2 in README
    cols = ['Depth','Area','ID','RAdeg','DEdeg','Seq','Hmag','ClSph','ClDk',
            'ClIr','ClPS','ClUn','ClInt','C0P0','C1P0','C2P0','C0P1','C1P1',
            'C2P1','C0P2','C1P2','C2P2','q_Bl','q_img','q_Cl','f_Vband',
            'f_zband','f_Jband','f_Tarms','f_Db','f_asym','f_Sarms','f_Bar',
            'f_PSCont','f_Edge-on','f_Face-on','f_Tp','f_Chain','f_Dk',
            'f_Bg','IDCL','Comm']
    
    
    # bring in data
    tbl2 = pd.read_csv(tmp_file, delim_whitespace=True, names=cols)
    
    #clean up
    os.remove(tmp_file)    
    
    #filter out all but the morphology classes
    morph_cols = ['Depth','ID','ClSph','ClDk','ClIr','ClPS','ClUn']
    tbl2 = tbl2.loc[:, morph_cols]
    
    # reduce votes for particular class by dividing by the total number 
    # classes voted for
    tbl2['total'] = tbl2['ClSph'] + tbl2['ClDk'] + tbl2['ClIr']\
                    + tbl2['ClPS'] + tbl2['ClUn']

    total = len(tbl2)
    for i in range(total):
        vals = tbl2.iloc[i,2:].values
        while (vals>0).sum() > 2:
            vals[vals==vals.min()]=0.0
            
        tbl2.iloc[i,2:] = vals


    for col in morph_cols[2:]:
        tbl2[col] /= tbl2['total']
        
    tbl2.drop('total', 1, inplace=True)

    
    # group the data by ID so that we can tally per source
    # TODO for now we only care about 2epoch so filter to that, in the future
    #      make it so that we can choose?
    grps = tbl2.loc[tbl2['Depth']=='2epoch', morph_cols[1:]].groupby('ID')
    
    new_tbl = pd.DataFrame(columns=morph_cols[1:])
    
    for name, group in grps:
        new_row = group.mean()
        new_row['ID'] = name
    
        new_tbl = new_tbl.append(new_row, ignore_index=True)
        
    if save_new_file:
        new_tbl.to_csv(save_new_file)
    
    return new_tbl
    
