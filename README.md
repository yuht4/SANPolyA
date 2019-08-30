# SANPolyA: a deep learning method for identifying Poly(A) signals

Polyadenylation plays a regulatory role in transcription. The recognition of polyadenylation signal (PAS) motif sequence is an important step in polyadenylation. In the past few years, some statistical machine learning-based and deep learning-based methods have been proposed for PAS identification. Although these methods predict PAS with success, there is room for their improvement on PAS identification.

### Requirements
1. The tool runs on linux machine.
2. [Anaconda2-4.4.1](https://docs.anaconda.com/anaconda/install/linux/). 
3. After installing Anaconda2, please use DeepPASTA_env1.yml file to create DeepPASTA suitable environment. 
   Please run the following command to activate the suitable environment:

		conda env create -f DeepPASTA_env1.yml
		source activate DeepPASTA_env

   Note: If the anaconda environment creates problem, please follow the steps written in "Creating environment for DeepPASTA" section.
4. Please [**download**](https://www.cs.ucr.edu/~aaref001/DeepPASTA_site.html) the trained parameters and put them in respective folders. 

## PolyA site prediction
In order to predict polyA sites, please use SANPolyA.py of pSANPolyA. 
Sample input files are given in the sample_input directory. Commands to run the polyA site prediction tool:

USAGE

	cd SANPolyA
	python SANPolyA.py {OPTIONS}	


OPTIONS

	-testSeq <input_sequence_file>	A FASTA file that contains human genomic sequences of length 206 nts. 

	-testSS <input_RNA_secondary_structure_file>	An input file that contains the RNA secondary structures of the input sequences.
					The tool expects three most energy efficient RNA secondary structures for each input sequence.
					These RNA secondary structures are generated using [RNAshapes](https://academic.oup.com/bioinformatics/article/22/4/500/184565).

	-o <output_file_name>		Output file name is given using this option. If this option is not used then the tool outputs
					AUC and AUPRC values of the prediction. In order to calculate the AUC and AUPRC values the tool 
					needs ground truth data. The ground truth data is added at the end of the title of each
					sequence. E.g. for a positive sequence example, the title is >chr15_100354095_positive_1; on
					the other hand, the title of a negative sequence example is >chr15_100565120_positive_0. 

EXAMPLE

	python DeepPASTA_polyA_site_prediction_testing.py -testSeq sample_sequence_input.hg19.fa -testSS sample_secondary_structure_input.txt  


### Input and output file of the polyA site prediction model
The model takes two files as input: sequence and RNA secondary structure files. The sequence file is a FASTA file that contains two lines per example.
The first line is the title of the sequence and the second line is the 200 nts sequence. The RNA secondary structure has four lines per example.
The first line is the title and the next three lines for three RNA secondary structures. The model outputs AUC and AUPRC values when -o option
is not used. In order to get the AUC and AUPRC values, the user must give the ground truth values using the title. E.g. title_ground_truth_value; 
for a positive sequence example, the title is >chr15_100354095_positive_1; on the other hand, the title of a negative sequence example is >chr15_100565120_positive_0. 
If the user uses -o option, the model will output the predicted likelihood values in an output file. 

 
Note: If you have any question or suggestion please feel free to email: yuht4@outlook.com
