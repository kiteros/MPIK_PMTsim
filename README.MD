# Muon tagging and electronics simulation

Scripts for SWGO gamma hadron seperation using muon tagging.

The following scripts provide command line interfaces:
* extract_data  
    Use to extract data from shower simulations. Simply give the path to the directory where the simulations are stored:  
    `python3 extract_data.py -e -p data/simulation/`  
    The script creates two files in the simulation directory containing all relevant data.
* build_tagger  
    Use to make simple muon taggers. Only charge and time taggers can be made. Uses data from extract_data.
* tag_events  
    Plot ROC curves and analyse shower tagging performance. To plot muon tagging ROC curve use:  
    `python3 tag_events.py -r data/simulation/`  
    To plot shower tagging ROC curves use (vary muon score cut `c` based on area of interest):  
    `python3 tag_events.py -a --sroc -c 1 2 -b 2 data/simulation/`

Ready to use functions can be found here:
* electronics_simulation  
    complete class for electronics simulation
* muon_tagging  
    classes for muon taggers
* shower_analysis  
    utility and plotting functions

The following taggers are available:
* *MuTagLR* charge distribution LR tagger  
    this tagger is constructed in *mu_tag_LR.py* with anlysis plots in *quick_hist.py*
* *MuTagRise* rise time LR tagger  
    this tagger is constructed in *rise_time_study.py*
* *MuTagChargeRise* combined time and charge LR tagger  
    this tagger is constructed in *time_plots.py*  
* *MuTagML* time and charge ML tagger (neural net)  
    this tagger is constructed in *mu_tag_ML.py*

Use *compare_methods.py* to compare different muon taggers.
See *esim_plots.py* for plots regarding electronics simulation.
