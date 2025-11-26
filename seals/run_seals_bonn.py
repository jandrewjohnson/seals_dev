import os
import sys

import hazelbean as hb
import pandas as pd

from seals import seals_generate_base_data, seals_initialize_project, seals_main, seals_process_coarse_timeseries, seals_tasks, seals_visualization_tasks



def convert_cgebox_output_to_seals_regional_projections_input(p):
    
    input_path = p.regional_projections_input_path
    output_path = os.path.join(p.cur_dir, 'regional_projections_input_pivoted.csv')
    p.regional_projections_input_override_path = output_path
    
    if p.run_this:
        
        if not hb.path_exists(output_path):
            df = hb.df_read(input_path)
            
            # Step 1: Melt the DataFrame to convert year columns into rows.

            # Get the list of columns to unpivot (years)
            years_to_unpivot = [col for col in df.columns if col.isdigit()]
            # years_to_unpivot = []
            melted = df.melt(
                id_vars=[p.regions_column_label, 'LandCover'], # Assumes the land cover column is named 'LandCover'
                value_vars=years_to_unpivot,
                var_name='year',
                value_name='value'
            )

            # Step 2: Pivot the melted DataFrame.
            # We set the region and year as the new index, and create new columns from 'LandCover' categories.
            merged_pivoted = melted.pivot_table(
                index=[p.regions_column_label, 'year'],
                columns='LandCover',
                values='value'
            ).reset_index()

            
            # Now add nuts_id
            merged_pivoted['nuts_id'], unique_countries = pd.factorize(merged_pivoted[p.regions_column_label])
            
            # Write a new file in the task dir and reassign the project attribute to the new csv
            hb.df_write(merged_pivoted, output_path)
        
        
def build_bonn_task_tree(p):

    # Define the project AOI
    p.project_aoi_task = p.add_task(seals_tasks.project_aoi)
    p.convert_cgebox_output_to_seals_regional_projections_input_task = p.add_task(convert_cgebox_output_to_seals_regional_projections_input)
    

    ##### FINE PROCESSED INPUTS #####
    p.fine_processed_inputs_task = p.add_task(seals_generate_base_data.fine_processed_inputs)
    p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels, parent=p.fine_processed_inputs_task, creates_dir=False)
    p.lulc_clip_task = p.add_task(seals_generate_base_data.lulc_clip, parent=p.fine_processed_inputs_task, creates_dir=False)
    p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications, parent=p.fine_processed_inputs_task, creates_dir=False)
    p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries, parent=p.fine_processed_inputs_task, creates_dir=False)
    p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions, parent=p.fine_processed_inputs_task, creates_dir=False)

    ##### COARSE CHANGE #####
    p.coarse_change_task = p.add_task(seals_process_coarse_timeseries.coarse_change, skip_existing=0)
    p.extraction_task = p.add_task(seals_process_coarse_timeseries.coarse_extraction, parent=p.coarse_change_task, run=1, skip_existing=0)
    p.coarse_simplified_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_proportion, parent=p.coarse_change_task, skip_existing=0)
    p.coarse_simplified_ha_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha, parent=p.coarse_change_task, skip_existing=0)
    p.coarse_simplified_ha_difference_from_previous_year_task = p.add_task(seals_process_coarse_timeseries.coarse_simplified_ha_difference_from_previous_year, parent=p.coarse_change_task, skip_existing=0)

    ##### REGIONAL
    p.regional_change_task = p.add_task(seals_process_coarse_timeseries.regional_change)

    ##### ALLOCATION #####
    p.allocations_task = p.add_iterator(seals_main.allocations, skip_existing=0)
    p.allocation_zones_task = p.add_iterator(seals_main.allocation_zones, run_in_parallel=p.run_in_parallel, parent=p.allocations_task, skip_existing=0)
    p.allocation_task = p.add_task(seals_main.allocation, parent=p.allocation_zones_task, skip_existing=0)

    ##### STITCH ZONES #####
    p.stitched_lulc_simplified_scenarios_task = p.add_task(seals_main.stitched_lulc_simplified_scenarios)

    ##### VIZUALIZE EXISTING DATA #####
    p.visualization_task = p.add_task(seals_visualization_tasks.visualization)
    p.lulc_pngs_task = p.add_task(seals_visualization_tasks.lulc_pngs, parent=p.visualization_task)




main = ''
if __name__ == '__main__':

    ### ------- ENVIRONMENT SETTINGS -------------------------------
    # Users should only need to edit lines in this section

    # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p = hb.ProjectFlow()

    # Assign project-level attributes to the p object (such as in p.base_data_dir = ... below)
    # including where the project_dir and base_data are located.
    # The project_name is used to name the project directory below. If the directory exists, each task will not recreate
    # files that already exist.
    p.user_dir = os.path.expanduser('~')
    p.extra_dirs = ['Files', 'seals', 'projects']
    p.project_name = 'bonn_esm_cgebox_seals'
    # p.project_name = p.project_name + '_' + hb.pretty_time() # If don't you want to recreate everything each time, comment out this line.

    # Based on the paths above, set the project_dir. All files will be created in this directory.
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)

    p.run_in_parallel = 1 # Must be set before building the task tree if the task tree has parralel iterator tasks.

    # Build the task tree via a building function and assign it to p. IF YOU WANT TO LOOK AT THE MODEL LOGIC, INSPECT THIS FUNCTION
    build_bonn_task_tree(p)

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    # Additionally, if you're clever, you can move files generated in your tasks to the right base_data_dir
    # directory so that they are available for future projects and avoids redundant processing.
    # The final directory has to be named base_data to match the naming convention on the google cloud bucket.
    p.base_data_dir = os.path.join(p.user_dir, 'Files/base_data')

    # ProjectFlow downloads all files automatically via the p.get_path() function. If you want it to download from a different
    # bucket than default, provide the name and credentials here. Otherwise uses default public data 'gtap_invest_seals_2023_04_21'.
    p.data_credentials_path = None
    p.input_bucket_name = None

    ## Set defaults and generate the scenario_definitions.csv if it doesn't exist.
    # SEALS will run based on the scenarios defined in a scenario_definitions.csv
    # If you have not run SEALS before, SEALS will generate it in your project's input_dir.
    # A useful way to get started is to to run SEALS on the test data without modification
    # and then edit the scenario_definitions.csv to your project needs.   
    p.scenario_definitions_filename = 'standard_scenarios.csv' 
    p.scenario_definitions_path = os.path.join(p.input_dir, p.scenario_definitions_filename)
    seals_initialize_project.initialize_scenario_definitions(p)
        
    # Set processing resolution: determines how large of a chunk should be processed at a time. 4 deg is about max for 64gb memory systems
    p.processing_resolution = 1.0 # In degrees. Must be in pyramid_compatible_resolutions

    seals_initialize_project.set_advanced_options(p)

    p.L = hb.get_logger('test_run_seals')
    hb.log('Created ProjectFlow object at ' + p.project_dir + '\n    from script ' + p.calling_script + '\n    with base_data set at ' + p.base_data_dir)

    p.execute()

    result = 'Done!'

