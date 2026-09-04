import requests
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import data_utils
from rdkit import Chem


def choose_smiles(smiles_entries):
    """
    Collapse a peptide's list of SMILES entries into a single canonical SMILES.

    DBAASP records multiple SMILES per peptide as tautomer/protonation variants
    of the *same* molecule (e.g. His HID/HIE, Arg C1/C2), so we pick one
    deterministically (preferring the PubChem-matched form) and canonicalize the
    atom ordering with RDKit.

    We deliberately use plain MolToSmiles rather than a tautomer canonicalizer:
    RDKit's TautomerEnumerator.Canonicalize strips stereochemistry, which would
    erase the D/L amino-acid distinction (D-amino acids are the whole point of
    keeping SMILES). Plain canonicalization preserves every stereocenter.

    Returns the canonical SMILES string, or "" if none is usable.
    """
    if not smiles_entries:
        return ""
    # prefer the PubChem-matched reference form, else the first entry
    entry = next((s for s in smiles_entries if s.get("pubChemCid")), smiles_entries[0])
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        return entry["smiles"]  # keep the raw string if RDKit cannot parse it
    return Chem.MolToSmiles(mol)

"""
DBAASP Open API v3
Servers: https://dbaasp.org/peptides

First, identify if {id} exists in the database,
Second, if exists, will get more details

If more than 100 consecutive invalid id is entered, 
it will consider as there is no more data to read

It will use sequence as a key to each row, pandas will remove duplicate
"""

class DBAASP:

    """
    Process DBAASP database 

    Attributes
    -----------------------------------------------    
    id_path, string 
        path to the dbaasp id, if not exists, it will automatically download
    detail_path, string
        path to the dbaasp details, if not exists, it will automatically download
    batch_size, int
        batch_size for multiprocessing
    """
    def __init__(self, id_path = None, detail_path = None, batch_size = 10):
        self.id_path = id_path
        self.detail_path = detail_path
        self.dbaasp_id = np.array([])      
        print(cpu_count())

        # if None for id_path
        if self.id_path == None:
            self.id_path = "dataset/data/dbaasp/dbaasp_id.txt"
            self.get_id_multiprocessing(batch_size = batch_size)
            print("Finished obtaining Monomer IDs in the DBAASP database")

        else:
            self.load_dbaasp_id()

        # if None for detail_path
        if self.detail_path == None:
            self.detail_path = "dataset/data/dbaasp/dbaasp_new_info.csv"
            self.get_details_multiprocessing(batch_size = batch_size)

    def load_dbaasp_id(self):
        """
        load existing id lists specified by the user

        Parameters
        -----------------------------------------------
        None

        Returns
        -----------------------------------------------
        None

        """
        self.dbaasp_id = np.loadtxt(fname=self.id_path, dtype=str)

    def dbaasp_api_get_id(self, batch_size = 10, id_start = 0, seq_len = 64):
        """
        using dbaasp api, get id

        Parameters
        -----------------------------------------------
        batch_size, int
            set batch size for multiprocessing
        id_start, int
            specify the start of the id for each thread
        seq_len, int
            specify the length of the sequence

        Returns
        -----------------------------------------------
        id, list
            id extracted for each thread
        """
        id = np.array([])
        for peptides_id in range(id_start, id_start + batch_size):
            peptides_id_info = requests.get("https://dbaasp.org/peptides", {"id.value" : str(peptides_id)})
            peptides_id_info = peptides_id_info.json()
            try:
                # the result of GET request will have "totalCount" key,
                # if 1, it means data exists,
                if peptides_id_info['totalCount'] > 0:
                    # if sequence is not empty
                    if peptides_id_info['data'][0]['sequence'] != "" and peptides_id_info['data'][0]['sequence'] is not None:

                        # less than specified length
                        if int(peptides_id_info['data'][0]['sequenceLength']) <= seq_len and peptides_id_info['data'][0]['complexity'] == "monomer":
                            id = np.append(id, np.array([peptides_id_info['data'][0]['dbaaspId']]))
                elif peptides_id_info['totalCount'] < 0:
                    dbaasp_log = open("log/dbaasp/dbaasp_peptides_id.log","a")
                    dbaasp_log.write(f"peptides id {peptides_id} has a negative count")
                    dbaasp_log.close()
            except:
                continue
        return id
    def convert_units_filter(self, peptide_id,activity, sequence):
        non_determine = False
        try: 
            # if , or empty space in the value
            # , -> considered as .
            #   -> considered as no space  
            conc = activity['concentration'].lstrip('><=-+–≥').rstrip('><=-+–≥')
            if ',' in conc:
                conc = conc.replace(',', '.')
            if ' ' in conc:
                conc = conc.replace(' ', '')
            
            # if it is in range
            # find the average of two values
            if '-' in conc or '–' in conc:
                target_concentration = np.array(conc.split('-'))
                if len(target_concentration) == 1:
                    target_concentration = np.array(conc.split('–'))

                target_concentration[0] = target_concentration[0].lstrip('><=-+–≥').rstrip('><=-+–≥')
                target_concentration[1] = target_concentration[1].lstrip('><=-+–≥').rstrip('><=-+–≥')

                tar_concentration = np.sum(target_concentration.astype(np.float32)) / 2.0
            
            elif 'upto' in conc:
                tar_concentration = float(conc.replace('upto', ''))
            # if it is in plus-minus
            # just remove plus-minus
            elif '±' in conc:
                tar_concentration = float(conc.split('±')[0])
            elif '\x07' in conc:
                tar_concentration = float(conc.split('\x07')[0])

            # very rare exception
            elif (conc.count(".") > 1) \
                or conc == '' \
                or "–>" in conc \
                or 'NA' in conc:
                non_determine = True 
            
            else:
                tar_concentration = float(conc)
        except Exception as e:
            print(e)
            print("VALUE FILTERING ERROR")
            print(peptide_id)
            non_determine = True 
        finally:
            # unit
            try:
                sequence = sequence.upper()
                sequence = sequence.lstrip(' ').rstrip(' ').replace(' ', '')
                if activity['unit'] is not None and (activity['saltType'] is None and activity['saltType'] == ''):
                    if activity['unit']['name'] == "µM" or activity['unit']['name'] =="uM" or activity['unit']['name'] =="μM" or activity['unit']['name'] == "µm" or activity['unit']['name'] == "μmol/L" or activity['unit']['name'] == "µmol/L":
                        
                        tar_concentration= data_utils.uM_to_ug_per_ml (tar_concentration, sequence)
                        
                    elif activity['unit']['name'] == "µg/ml" or activity['unit']['name'] == "μg/ml" or activity['unit']['name'] == "µg/mL" or activity['unit']['name'] == "μg/mL" or activity['unit']['name'] == "ug/ml" or activity['unit']['name'] == "μg /ml ":
                        tar_concentration = tar_concentration
                    elif activity['unit']['name'] == "nM":
                        tar_concentration = data_utils.uM_to_ug_per_ml (tar_concentration, sequence) / 1000
                    elif activity['unit']['name'] == "mM" or activity['unit']['name'] == "mm" or activity['unit']['name'] == "microM":
                        tar_concentration = data_utils.uM_to_ug_per_ml (tar_concentration, sequence) * 1000  
                    elif activity['unit']['name'] == "ng/ml":
                        tar_concentration = tar_concentration / 1000
                    elif activity['unit']['name'] == "mg/L":
                        tar_concentration = tar_concentration * 1000
                    elif activity['unit']['name'] == "g/L" or activity['unit']['name'] == "g/L":
                        tar_concentration = tar_concentration * 1000000
                    else:
                        non_determine = True
                else:
                    return None
            except Exception as e:
                print(tar_concentration)
                print(e)
                print("UNIT CONVERSION ERROR")
                print(peptide_id)
                non_determine = True 
            finally:
                if non_determine:
                    return None
                else:
                    return tar_concentration
            

    def dbaasp_api_get_details(self, batch_size = 10, ids = None):
        """
        get details from dbaasp api

        Parameters
        -----------------------------------------------
        batch_size, int
            set batch size for multiprocessing
        ids, list
            id list from dbaasp_api_get_id

        Returns
        -----------------------------------------------
        details, list
            details in list
        """

        name = []
        sequence = []
        smiles = []

        nTerminus = []
        cTerminus = []

        targetGroups = []
        targetObjects = []

        targetActivities = []

        toxicities = []
        unusuals = []

        if ids is not None:

            for peptides_id in ids:
                try:
                    peptides_info = requests.get(f"https://dbaasp.org/peptides/{peptides_id}")
                    peptides_info = peptides_info.json()

                    # cleaned sequence (case preserved: DBAASP encodes D-amino
                    # acids as lowercase letters)
                    temp_sequence = (peptides_info['sequence'] or "").replace(' ', '')

                    # canonical SMILES from DBAASP
                    temp_smiles = choose_smiles(peptides_info.get('smiles'))

                    # peptides without a DBAASP SMILES are marked failed and skipped
                    if not temp_smiles:
                        dbaasp_log = open("log/dbaasp/dbaasp.log","a")
                        dbaasp_log.write(f"NOT SAVED\t{peptides_id}\t{temp_sequence}\tno SMILES in DBAASP\n")
                        dbaasp_log.close()
                        continue

                    # Parse every field into locals first; only append to the
                    # output columns once all parsing has succeeded, so a failure
                    # midway never leaves the columns at unequal lengths.

                    # nTerminus and cTerminus
                    if peptides_info['nTerminus'] is not None:
                        temp_nTerminus = peptides_info['nTerminus']['name']
                    else:
                        temp_nTerminus = None

                    if peptides_info['cTerminus'] is not None:
                        temp_cTerminus = peptides_info['cTerminus']['name']
                    else:
                        temp_cTerminus = None

                    # target groups
                    target_groups = []
                    for target_group in peptides_info['targetGroups']:
                        target_groups.append(target_group['name'])

                    # target
                    targets = []
                    for target_object in peptides_info['targetObjects']:
                        targets.append(target_object['name'])

                    species_list = []
                    # target species
                    for id_idx, species_info in enumerate(peptides_info['targetActivities']):
                        species = dict()

                        if species_info['activityMeasureValue'].lstrip().startswith('IC') \
                        or species_info['activityMeasureValue'].lstrip().startswith('MIC'):

                            concentration = self.convert_units_filter(peptides_info['dbaaspId'], species_info, peptides_info['sequence'])
                            if concentration is None:
                                pass
                            else:
                                species_name = species_info['targetSpecies']['name']

                                species_measure = species_info['activityMeasureValue']

                                unit = 'µg/ml'


                                species = {'species_name':species_name,
                                        'species_measure':species_measure,
                                        'concentration':concentration,
                                        'unit':unit}

                                species_list.append(species)

                    toxicity_list = []
                    for id_idx, toxicity_activity in enumerate(peptides_info['hemoliticCytotoxicActivities']):

                        toxicity = dict()
                        if toxicity_activity['concentration'] is not None and toxicity_activity['unit'] is not None:
                            concentration = self.convert_units_filter(peptides_info['dbaaspId'], toxicity_activity, peptides_info['sequence'])
                            if concentration is not None:
                                toxic_value = concentration
                                toxic_target = toxicity_activity['targetCell']['name']
                                toxic_measure = toxicity_activity['activityMeasureForLysisValue']
                                toxic_unit = 'µg/ml'

                                toxicity = {'toxic_target':toxic_target,
                                            'toxic_measure':toxic_measure,
                                            'toxic_value':toxic_value,
                                            'toxic_unit':toxic_unit}

                                toxicity_list.append(toxicity)

                    unusual_list= []

                    for id_idx, unusual_activity in enumerate(peptides_info['unusualAminoAcids']):

                        unusual = dict()
                        if unusual_activity['position'] is not None and unusual_activity['modificationType'] is not None:

                            if unusual_activity['modificationType']['name'] is not None:

                                unusual = {'position':unusual_activity['position'],
                                            'name':unusual_activity['modificationType']['name']}

                                unusual_list.append(unusual)

                    # all parsing succeeded -- commit this peptide to every column
                    smiles.append(temp_smiles)
                    sequence.append(peptides_info['sequence'])
                    name.append(peptides_info['name'])
                    nTerminus.append(temp_nTerminus)
                    cTerminus.append(temp_cTerminus)
                    targetGroups.append(target_groups)
                    targetObjects.append(targets)
                    targetActivities.append(species_list)
                    toxicities.append(toxicity_list)
                    unusuals.append(unusual_list)
                except Exception as e:
                    dbaasp_log = open("log/dbaasp/dbaasp.log","a")
                    dbaasp_log.write(f"NOT SAVED\t{peptides_id}\t{locals().get('temp_sequence', '?')}\terror: {e}\n")
                    dbaasp_log.close()
                else:
                    pass

        return [name, sequence, nTerminus, cTerminus, targetGroups, targetObjects, targetActivities, toxicities, unusuals, smiles]

    def get_id_multiprocessing(self, batch_size):
        still_exist = True
        BATCH_SIZE  = batch_size
        start_multiprocessing = 0
        dbaasp_id = np.array([])
        while still_exist:
            # send batch requests by multiprocessing
            with Pool(processes=cpu_count()) as pl:
                results = pl.starmap(self.dbaasp_api_get_id, 
                                    [(BATCH_SIZE, start) for start in range(start_multiprocessing, 
                                                                            start_multiprocessing + cpu_count() * BATCH_SIZE, 
                                                                            BATCH_SIZE)])
            start_multiprocessing = start_multiprocessing + cpu_count() * BATCH_SIZE
            previous_count = len(dbaasp_id)
            for result in results:
                dbaasp_id = np.append(dbaasp_id, result)
            
            if previous_count == len(dbaasp_id):
                still_exist = False

        dbaasp_save = open(self.id_path, "w")
        for id in dbaasp_id:
            dbaasp_save.write(f"{id}\n")

        self.dbaasp_id = dbaasp_id
        dbaasp_save.close()

    def get_details_multiprocessing(self, batch_size):
        still_exist = True
        BATCH_SIZE  = batch_size
        start_multiprocessing = 0

        dbaasp_details = {"name":[], "sequence": [], "smiles": [], "nTerminus" : [], "cTerminus" : [], "targetGroups": [], "targetObjects": [], "targetActivities": [], "toxicities": [], "unusuals":[]}
        batch = []


        for i in range(0, len(self.dbaasp_id), BATCH_SIZE):
            batch.append(self.dbaasp_id[i : i + BATCH_SIZE])

        while still_exist:
            end = 0
            if start_multiprocessing + cpu_count() > len(batch):
                end = len(batch)
                still_exist = False
            else:
                end = start_multiprocessing + cpu_count()

            # send batch requests by multiprocessing
            with Pool(processes=cpu_count()) as pl:
                results = pl.starmap(self.dbaasp_api_get_details, 
                                    [(BATCH_SIZE, batch[i]) for i in range(start_multiprocessing, end)])
                
            start_multiprocessing = start_multiprocessing + cpu_count()

            for result in results:
                dbaasp_details["name"] = dbaasp_details["name"] + result[0]
                dbaasp_details["sequence"] = dbaasp_details["sequence"] + result[1]
                dbaasp_details["smiles"] = dbaasp_details["smiles"] + result[9]
                dbaasp_details["nTerminus"] = dbaasp_details["nTerminus"] + result[2]
                dbaasp_details["cTerminus"] = dbaasp_details["cTerminus"] + result[3]
                dbaasp_details["targetGroups"] = dbaasp_details["targetGroups"] + result[4]
                dbaasp_details["targetObjects"] = dbaasp_details["targetObjects"] + result[5]
                dbaasp_details["targetActivities"] = dbaasp_details["targetActivities"] + result[6]
                dbaasp_details["toxicities"] = dbaasp_details["toxicities"] + result[7]
                dbaasp_details["unusuals"] = dbaasp_details["unusuals"] + result[8]

            
            
        filtered = pd.DataFrame(dbaasp_details)

        # dedup key: canonical SMILES when present (captures D-amino acids and
        # terminal modifications that a case-preserved sequence cannot
        # distinguish), otherwise fall back to the sequence itself
        filtered["_dedup_key"] = filtered["smiles"].where(
            filtered["smiles"].astype(bool), filtered["sequence"]
        )
        filtered = filtered.drop_duplicates(subset="_dedup_key").drop(columns="_dedup_key")

        filtered.to_csv(self.detail_path, index=False)
                                    
