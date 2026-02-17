def diffusion_bootstrap(original_matrix, iterations):
    """
    ########## Definition Inputs ##################################################################################################################
    original_matrix         : Sorted diffusion data (5D - [rows, columns, slices, directions, averages]).
    ########## Definition Outputs #################################################################################################################
    bootstrap_matrix        : Sorted bootstrapped diffusion data (5D - [rows, columns, slices, directions, bootstrap iterations]).
    ########## Definition Information #############################################################################################################
    ### Written by Tyler E. Cork, tyler.e.cork@gmail.com
    ### Cardiac Magnetic Resonance (CMR) Group, Leland Stanford Jr University, 2022
    """
    ########## Import modules #####################################################################################################################
    import numpy as np                                                                                                 # Import numpy module
    import random                                                                                                      # Import random module
    ########## Extract Matrix Size ################################################################################################################
    rows       = original_matrix.shape[0]                                                                              # Extract number of rows
    columns    = original_matrix.shape[1]                                                                              # Extract number of columns
    slices     = original_matrix.shape[2]                                                                              # Extract number of slices
    directions = original_matrix.shape[3]                                                                              # Extract number of directions
    averages   = original_matrix.shape[4]                                                                              # Extract number of averages
    ########## Store Direction and Average Information ############################################################################################
    dirs_vs_avgs_matrix = np.zeros((averages, directions))                                                             # Initialize direcions vs averages matrix
    for avgs in range(averages):                                                                                       # Iterate through averages
        for dirs in range(directions):                                                                                     # Iterate through directions
            avg_term                        = avgs + 1                                                                          # Define average term
            dir_term                        = (dirs + 1) / 1000                                                                 # Define direction term
            end_term                        = 4 / 1000                                                                          # Define end term
            dirs_vs_avgs_matrix[avgs, dirs] = np.round((avg_term + dir_term + end_term), 3)                                     # Store direction and average information in direcions vs averages matrix
    ########## Randomly Bootstrap Unique Datasets for each Iteration  #############################################################################
    permentation = averages ** directions                                                                               # Calculate permutation of data
    if iterations > permentation:                                                                                       # If iterations is greater than permutation ...
        iterations = permentation                                                                                            # Redefine number of iterations
        print('Desired iterations is not possible with current dataset.')                                                    # Print warning
        print('Iterations has been set to the permutation of directions and averages with replacment.')                      # Print warning
    print('Number of iterations to be performed: %i' %int(iterations))                                                   # Print warning
    iterations_matrix = np.zeros((iterations + 1, directions))                                                           # Initalize iteration matrix with one extra iteration
    for iters in range(iterations):                                                                                      # Iterate through iterations
        while iterations_matrix[iters, :].tolist() in iterations_matrix.tolist():                                            # While current iteration is in iteration maxtrix ...
            current_iteration = np.zeros((1, directions))                                                                        # Ititalize current iteration
            for dirs in range(directions):                                                                                       # Iterate through directions
                random_avgerage            = random.randint(0, averages - 1)                                                         # Select random average
                current_iteration[0, dirs] = dirs_vs_avgs_matrix[random_avgerage, dirs]                                              # Set current iteration with random average
            current_iteration_list = current_iteration.tolist()                                                                  # Convert current iteration to list
            iterations_list        = iterations_matrix.tolist()                                                                  # Convert iterations matrix to list
            check_var              = any(item in current_iteration_list for item in iterations_list)                             # Check variable for current iteration list in iterations list
            if not check_var:                                                                                                # If check variable is false ... 
                iterations_matrix[iters, :] = current_iteration[0, :]                                                            # Store current iteration in iteration matrix
                break                                                                                                            # Break while loop to continue to next iteration
    iterations_matrix = iterations_matrix[:-1, :]                                                                          # Remove extra iteration from iteration matrix
    ########## Extract Bootstrap Averages #########################################################################################################
    # directions_matrix = np.zeros((directions, iterations))                                                                # Initialize direction matrix (Removed: not necessary)
    avgerage_matrix   = np.zeros((directions, iterations))                                                                 # Initialize average matrix
    for iters in range(iterations):                                                                                        # Iterate through iterations
        for dirs in range(directions):                                                                                         # Iterate through directions
    #         dir_idx                     = int(float('.' + str(iters_matrix[iters, dirs]).split('.')[-1]) * 100 - 1)               # Extract direction index from iteration matrix (Removed: not necessary)
    #         dir_matrix[dirs, iters]     = int(dir_idx)                                                                            # Store direction index in direction matrix (Removed: not necessary)
            avgerage_index               = int(float(str(iterations_matrix[iters, dirs]).split('.')[0]) - 1)                       # Extract average index from iteration matrix
            avgerage_matrix[dirs, iters] = int(avgerage_index)                                                                     # Store average index in average matrix
    ########## Store Bootstrap Matrix #############################################################################################################            
    bootstrap_matrix = np.zeros([rows, columns, slices, directions, iterations])                                           # Initialize bootstrap matrix
    for iters in range(iterations):                                                                                        # Iterate through iterations
        for dirs in range(directions):                                                                                         # Iterate through directions
            bootstrap_matrix[:, :, :, dirs, iters] = original_matrix[:, :, :, dirs, int(avgerage_matrix[dirs, iters])]             # Store original matrix in bootstrap matrix
    return bootstrap_matrix