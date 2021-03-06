class Constants:
    min_comparator_datasets: int = 30
    structure_regex: str = "*.pdb"
    reflections_regex: str = "*.mtz"
    smiles_regex: str = "*.smiles"

    structure_factors: str = "FWT,PHWT"

    residue_names = ""

    affinity_event_map_res_path: str = "{dtag}_{model}_{chain}_{insertion}.ccp4"
    affinity_event_map_pos_path: str = "{dtag}_{x}_{y}_{z}.ccp4"


    @staticmethod
    def update(dictionary):
        for key, value in dictionary.items():
            Constants.__dict__[key] = value

