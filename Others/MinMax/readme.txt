Reference files and python code to run Min/Max QC test
======================================================

  The MinMax QC test checks if a given temperature/salinity observation lays inside a local validity interval estimated from an historical dataset with specific quality check applied. Instead of estimating the validity interval bounds from first and second order statistical moments, i.e. mean and variance, the local minimum and maximum values are used, allowing to account for the non-gaussianity of the local parameter distribution. 

  The present package contains :
  ▪ Le notebook /.../predictions_minmax_comparation.ipynb sélectionne cinq profils représentatifs pour chaque catégorie 
    de prédiction (TP, TN, FP et FN), télécharge les profils correspondants depuis le GDAC Argo, 
    puis compare leurs mesures de température et de salinité aux enveloppes Min/Max de référence 
    afin d’identifier les points situés en dehors des intervalles de validité. (/...//MinMax/lightgbm_predictions/predictions_minmax_comparation.ipynb)
    ▪ a directory with the reference Netcdf files: « GRID_MIN_MAX.nc », « TEMP_MIN_MAX.nc » and « PSAL_MIN_MAX.nc »
    ▪ a python function « Tools_MinMax_ISEA.py » with the tools required to use the reference files
    ▪ a python script « Example_MinMax_ISEA.py » as an example showing how to plot a profile and its Min/Max envelope
    ▪ a directory with auxiliary files : ISEA grid characteristics, T/S example profile
    
 
References: 
  Gourrion, J., Szekely, T., Killick, R., Owens, B., Reverdin, G. and Chapron, B. (2020). Improved Statistical Method for Quality Control of Hydrographic Observations. Journal of Atmospheric and Oceanic Technology, 37(5), 789-806. DOI: 10.1175/JTECH-D-18-0244.1  




