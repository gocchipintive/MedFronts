## Code of the manuscript: Fronts as drivers of phytoplankton community shifts in the Mediterranean Sea, Occhipinti, G. et al. 2025

Description of the python scripts:
- Analysis:
  - change_extension_nc.py: restrict netcdf file to the Mediterranean Sea extent
  - compute_chl_clim.py: compute surface chlorophyll climatology from model outputs
  - compute_chl_over_fronts.py: compute the chlorophyll averaged along latitude and along strong fronts, weak fronts, no fronts
  - compute_fronts_frequency.py: compute the frequency of strong fronts, weak fronts, no fronts in each ecoregion
  - compute_hi.py: compute the Hetereogeneity Index fom model sea temperature data
  - compute_hi_mpi.py: paralelized version with mpi of the previous script
  - compute_pft_clima_fronts.py: compute climatological average of each plankton functional type (PFT) in each ecoregion and along strong fronts, weak fronts, no fronts
  - compute_pft_increments.py: compute the percentual increments of each PFT in fronts with respect to no fronts, subdivided in ecoregions and seasons
  - compute_pft_longitudinal.py: compute each PFT chlorophyll averaged along latitude and along strong fronts, weak fronts, no fronts
- Plotting:
  -  fronts_over_regions.py: plot the frequency of fronts in each ecoregion
  -  plot_chl_fronts_longitudinal.py: plot the longitudinal distribution of chlorophyll concentration along fronts 
  -  plot_front_detection.py: plot of the methodology of front detection
  -  plot_fronts_freq.py: plot front frequency and climatological mean chlorophyll on a map of the Mediterranean Sea
  -  plot_pft_clima_fronts.py: plot the climatological year PFT chlorophyll subdivided in ecoregions and fronts
  -  plot_pft_longitudinal.py: plot the longitudinal distribution of PFT chlorophyll concentration
  -  plot_relative_pft.py: plot the relative concentration of PFTs subdivided in ecoregions and frons
  -  plot_table.py: plot tables showing the percentual increments of PFT and nutrients subdivided in fronts and seasons
