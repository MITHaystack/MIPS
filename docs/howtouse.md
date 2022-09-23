## How to Use

# Basic simulation

To simulate a single scenario at at one point of time and space the function `is_snr` in the `isr_performance.py` file will take in number of inputs and then output a set of numpy arrays and a dictionary for the expected error bars. This is the baseline function that performs the calculation of SNR, expected required measurement time and expected errors in the fitted parameters.

The function `simulate_data` in `isr_sim_array.py` will take in a series of dictionaries parameterizing a simulation over a parameter space and then output an [xarray dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) that can be saved as a netcdf file. The three input dictionaries are `data_dims` which hold information on the dimensions that will be simulated over, `coorddict` which will hold arrays of parameters that the simulation will be varied over, and `paramvalues` which holds other parameters that will be static during the simulation.

## Mapping

A common use case for MIPS is to study the performance of a radar system over a spatial area. This can be done using the `map_radar_array` function in `isr_mapper.py` which takes a number of input parameters can creates a performance simulation gridded over an area. The `map_radar_array` function uses `isr_array_sim` which performs various geometric calculations for multistatic operation. It is suggested that the `map_radar_array` is used as it is a better interface.

The grid data created from `map_radar_array` can be plotted using the `isr_map_plot` function in `isr_plotting.py`. This will output maps of the desired performance parameters.



A script called `runmapping.py` is can be used to create both mapping data and plots using configuration files. Samples of these config files are in `examples/yamlmappingexamples`.
