import pandas as pd
import argparse
import os
def dataframe_difference(df1, df2):
#    """Find rows which are different between two DataFrames."""
    list1=list(df1)
    list2=list(df2)
    return list(set(list1) & set(list2))


def common_cpgs(main_data_frame):
    # read in first dataframe
    data_repl_ann=pd.read_csv("./replc_dmrcate/replc_anno_fdr_0.05.csv")
    data_repl_dmrs=data_repl_ann[data_repl_ann["is.sig"]==True]
    data_repl_dmrs=data_repl_dmrs.reset_index()
    data_cpg_repl=data_repl_dmrs["names"]
    
    
    frame_disc_ann=pd.read_csv("./disc_dmrcate/disc_anno_FDR0.05.csv")
    frame_disc_dmrs=frame_disc_ann[frame_disc_ann["is.sig"]==True]
    frame_disc_dmrs=frame_disc_dmrs.reset_index()
    frame_cpg_disc=frame_disc_dmrs["names"]
    

    # get the sites from second dataframe which are present in the first dataframe 
    list_common=dataframe_difference(frame_cpg_disc, data_cpg_repl)
    
    # read in the main data frame to get the features
    features_extraction=pd.read_csv(main_data_frame,low_memory=False)
    
    #initialize keys to contain common cpgs
    keys=list_common
    # extract the initial features from the file
    initial_features=features_extraction.iloc[1:14,:]
    # get the common cpgs and its values into the table common_cpg_sites
    common_cpg_sites=features_extraction[features_extraction["Unnamed: 0"].isin(keys)]
    # final_extract contains the list of cpgs and initial features
    final_extract=initial_features.append(common_cpg_sites)
    final_extract=final_extract.reset_index()
    final_extract.drop("index",axis=1,inplace=True)
    final_extract=final_extract.reset_index()
    #take a transpose of the dataframe to aid in further analysis and for machine learning model
    
    features=final_extract.set_index('Unnamed: 0').transpose()
    features.rename(columns={"Unnamed: 0":"Samples"},inplace=True)
    features=features[1:]
    #copy into a file to get Intermediate csv file which we will remove.
    features.to_csv("./Intermediate.csv")

def final_processing(features,Prediction,Entire):
    #read in the Intermediate file
    CPGsites=pd.read_csv("./Intermediate.csv")
    # read the Metadata file obtained from the authors
    Samples_desc=pd.read_csv("./final_files/MetaData.xls",sep="\t")
    #Get a list of cols to remove which are common between the two
    list_of_cols=['Unnamed: 0','characteristics: age', 'characteristics: sex',
           'characteristics: race', 'characteristics: hiv',
           'characteristics: smoking', 'characteristics: ARTadherence',
           'characteristics: WBC_new', 'characteristics: CD8T',
           'characteristics: CD4T','characteristics: Gran', 'characteristics: NK','characteristics: Bcell', 'characteristics: Mono']
    for i in list_of_cols:
    	CPGsites.drop(i, axis=1, inplace=True)
    #join both the files to obtain one dataframe
    final_file=Samples_desc.join(CPGsites)
    #remove a list of columns not required
    list_drop_cols=['title', 'source name', 'organism', 'idat file',
           'idat file.1','characteristics: hiv',
         "label","description","platform"]

    for i in list_drop_cols:
            final_file.drop(i, axis=1, inplace=True)

    final_file.dropna(inplace=True)
    #Get one file which is contains both the prediction and features 
    final_file.to_csv(Entire,index=False)
    final_file.dropna(inplace=True)
    #To get two different files of Prediction and Features file
    pred=final_file[["Sample name","characteristics: vacsIndex"]]
    final_file.drop(["Sample name","characteristics: vacsIndex"],axis=1,inplace=True)
    pred.to_csv(Prediction,index=False)
    final_file.to_csv(features,index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-fcsv", "--final-csv-file" ,help="Path to the final csv file obtained from DMR cate analysis")
    parser.add_argument("-ofcsv", "--output-features-file" ,help="Path to the final output features csv file")
    parser.add_argument("-opcsv", "--output-Prediction-file" ,help="Path to the final output Prediction csv file")
    parser.add_argument("-oecsv", "--output-Entire-file" ,help="Path to the final output Entire csv file")
    args = vars(parser.parse_args())
    dataframe=args['final_csv_file']
    features=args['output_features_file']
    Prediction=args['output_Prediction_file']
    Entire=args['output_Entire_file']
    common_cpgs(dataframe)
    final_processing(features,Prediction,Entire)
    os.remove("./Intermediate.csv")
if __name__=="__main__":
    main()

