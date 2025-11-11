#!/usr/bin/env python3

import csv,os,gzip,shutil,glob,argparse
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Before run, get and save genome gff files (*.gff) and HMMer output (*.out) in corresponding subdirs under target_dir
# Usage: python generate_input.py -i /target_dir/

#--- Remove domain duplications based on e-values
def resolve_overlaps(df):
    grouped = df.groupby('ProteinID')
    resolved_rows = []

    for protein_id,group in grouped:
        domains = [(row['Ali_Coord_From'],row['Ali_Coord_To'],row) for _,row in group.iterrows()]

        domains.sort(key=lambda x: x[0])
        resolved = []
        for domain in domains:
            if not resolved or resolved[-1][1] < domain[0]:
                resolved.append(domain)
            else:
                last = resolved.pop()
                if domain[2]['E_Value'] < last[2]['E_Value']:
                    resolved.append(domain)
                else:
                    resolved.append(last)

        resolved_rows.extend([d[2] for d in resolved])
    return pd.DataFrame(resolved_rows)

#--- Generate files for tokenization
def domains_per_protein_with_sep(in_csv,out_txt,sep_token="[SEP]"):
    domains = defaultdict(list)
    with open(in_csv,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("ProteinID") or "").strip()
            dom = (row.get("Domain_Name") or "").strip()
            if not pid or not dom: continue
            try:
                start = int(row.get("Ali_Coord_From") or row.get("HMM_Coord_From") or 0)
            except ValueError:
                start = 0

            domains[pid].append((start,dom))

    # Write: one line per protein = "Dom1 Dom2 ... [SEP]"
    all_tokens = []
    for pid in sorted(domains.keys()):
        # sort by coordinate to ensure domain order on the protein
        doms_sorted = [d for _,d in sorted(domains[pid],key=lambda x: x[0])]
        if doms_sorted:
            all_tokens.extend(doms_sorted)
            all_tokens.append(sep_token)

    with open(out_txt,"w",encoding="utf-8") as out:
        out.write(" ".join(all_tokens) + "\n")


# Main
# argument setting
ap = argparse.ArgumentParser()
ap.add_argument("--target_dir","-i",type=str,required=True,help="Target directory name containing sub-directories with genomic gff and HMMer output files")
args = ap.parse_args()

subdirs = [os.path.join(args.target_dir,name) for name in os.listdir(args.target_dir) if os.path.isdir(os.path.join(args.target_dir,name))]
for subdir in tqdm(subdirs,desc="Processing sub-dirs"):
    gff_files = [name for name in os.listdir(subdir) if name.endswith('.gff')]
    out_files = [name for name in os.listdir(subdir) if name.endswith('.out')]
    for gff_file in gff_files:
        for out_file in out_files:
            gff_file_path = os.path.join(subdir,gff_file)
            output_file = os.path.join(subdir,out_file)
            
            domain_info = {}
            # Collect info from HMMer output files
            with open(output_file,'r') as f:
                for line in f:
                    if not line.startswith("#"):
                        fields = line.strip().split()
                        domain_name = fields[0]
                        domain_id = fields[1]
                        protein_id = fields[3]
                        E_value = float(fields[6])
                        c_Evalue = fields[11]
                        i_Evalue = fields[12]
                        hmm_coord_from = fields[15]
                        hmm_coord_to = fields[16]
                        ali_coord_from = fields[17]
                        ali_coord_to = fields[18]
                        env_coord_from = fields[19]
                        env_coord_to = fields[20]
                        accuracy = fields[21]

                        if protein_id not in domain_info and E_value < 0.01:
                            domain_info[protein_id] = []
                        
                        if E_value < 0.01:
                            domain_info[protein_id].append({
                                'domain_name': domain_name,
                                'domain_id': domain_id,
                                'E_value': E_value,
                                'c_Evalue': c_Evalue,
                                'i_Evalue': i_Evalue,
                                'hmm_coord_from': hmm_coord_from,
                                'hmm_coord_to': hmm_coord_to,
                                'ali_coord_from': ali_coord_from,
                                'ali_coord_to': ali_coord_to,
                                'env_coord_from': env_coord_from,
                                'env_coord_to': env_coord_to,
                                'accuracy': accuracy,
                            })

            protein_info_list = []
            
            # Collect protein info from GFF files
            with open(gff_file_path) as handle:
                for line in handle:
                    if line.startswith("#"): continue
                    seqid,source,feature_type,start,end,score,strand,phase,attributes = line.strip().split("\t")
                    attribute_dict = dict(item.split("=") for item in attributes.split(";"))

                    if 'protein_id' in attribute_dict:
                        protein_id = attribute_dict['protein_id']
                        product = attribute_dict['product']
                        start_position = int(start)
                        end_position = int(end)
                        strand_info = strand
                        protein_info_list.append({
                            'ProteinID': protein_id,
                            'ChrID': seqid,
                            'Product': product,
                            'Start_Position': start_position,
                            'End_Position': end_position,
                            'Strand': strand_info
                        })

            # Split data per seqid
            seqid_data = {}
            for protein_data in protein_info_list:
                seqid = protein_data['ChrID']
                if seqid not in seqid_data:
                    seqid_data[seqid] = []
                seqid_data[seqid].append(protein_data)

            # Output per ChrID
            for seqid,proteins in seqid_data.items():
                csv_file_path = os.path.join(f"{gff_file_path[:-4]}_{seqid}.csv")
                with open(csv_file_path,'w',newline='') as csvfile:
                    fieldnames = ['ProteinID','ChrID','Product','Start_Position','End_Position','Strand','Domain_Name','Domain_ID','E_Value','C_EValue','I_EValue','HMM_Coord_From','HMM_Coord_To','Ali_Coord_From','Ali_Coord_To','Env_Coord_From','Env_Coord_To','Accuracy']
                    writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
                    writer.writeheader()
                    for protein_data in proteins:
                        protein_id = protein_data['ProteinID']
                        for domain_data in domain_info.get(protein_id,[]):
                            writer.writerow({
                                'ProteinID': protein_id,
                                'ChrID': protein_data['ChrID'],
                                'Product': protein_data['Product'],
                                'Start_Position': protein_data['Start_Position'],
                                'End_Position': protein_data['End_Position'],
                                'Strand': protein_data['Strand'],
                                'Domain_Name': domain_data['domain_name'],
                                'Domain_ID': domain_data['domain_id'],
                                'E_Value': domain_data['E_value'],
                                'C_EValue': domain_data['c_Evalue'],
                                'I_EValue': domain_data['i_Evalue'],
                                'HMM_Coord_From': domain_data['hmm_coord_from'],
                                'HMM_Coord_To': domain_data['hmm_coord_to'],
                                'Ali_Coord_From': domain_data['ali_coord_from'],
                                'Ali_Coord_To': domain_data['ali_coord_to'],
                                'Env_Coord_From': domain_data['env_coord_from'],
                                'Env_Coord_To': domain_data['env_coord_to'],
                                'Accuracy': domain_data['accuracy']
                            })

    # Remove duplication
    csv_files = [file for file in os.listdir(subdir) if file.endswith('.csv') and not file.endswith(('_sorted.csv','_removed.csv','_counts.csv'))]
    for csv_file in tqdm(csv_files,desc="Processing CSV files",leave=False):
        full_csv_path = os.path.join(subdir,csv_file)
        data = pd.read_csv(full_csv_path)

        resolved_df = resolve_overlaps(data)
        if not "Start_Position" in resolved_df.columns:
            print(f"Skipping {full_csv_path}: no records found.")
            continue
        resolved_df = resolved_df.sort_values(by="Start_Position")

        output_file_name = os.path.basename(full_csv_path).replace('.csv','_removed.csv')
        output_path = os.path.join(subdir,output_file_name)
        resolved_df.to_csv(output_path,index=False,float_format="%.1e")


    # Generate files for tokenization
    csv_files = [file for file in os.listdir(subdir) if file.endswith('_removed.csv')]
    for csv_file in tqdm(csv_files,desc="Processing CSV files",leave=False):
        full_csv_path = os.path.join(subdir,csv_file)
        data = pd.read_csv(full_csv_path)

        oname = os.path.basename(full_csv_path).replace('.csv','.txt')
        opath = os.path.join(subdir,oname)
        domains_per_protein_with_sep(full_csv_path,opath)


#---end---

