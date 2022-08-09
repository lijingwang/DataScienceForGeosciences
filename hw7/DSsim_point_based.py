## DSsim_point_based - the Python 3 code package for direction sampling(DS). This DS method is point based.
## Authors: David Zhen Yin <yinzhen@stanford.edu>; Zuo Chen <chenzuo789@outlook.com>
## Reference: 
## Zuo, et al. (2020). A Tree-Based Direct Sampling Method for Stochastic Surface and Subsurface Hydrological Modeling. Water Resources Research, 56(2).
import numpy as np
#from tqdm import tqdm
import tqdm.notebook as tq

def DSsim_point_based(SimulationGrid,
                   TI,
                   DS_PatternRadius = 16,
                   DS_Neighbors = 30, 
                   DS_DistanceThreshold_factor = 0.05,
                   TI_SearchFraction = 0.2):
    '''
    This is the main function of point-based direct sampling. 
    Parameters - 
    SimulationGrid: simulation grid with hard data values, no-data area as np.nan, 2D array.
    TI: Trianining image, 2D array.
    DS_SearchingRadius: searching radius to to obtain the data pattern. 
    DS_Neighbors: maximum neighborhood hard data points in data searching pattern.
    DS_DistanceThreshold_factor: DS_Threshold = DS_DistanceThreshold_factor*(max(SimulationGrid) - min(SimulationGrid))
    TI_SearchFraction: fractions of TI to search when computing the distance, searching path is randomized.
    '''
    
    # specify the basic information about the simulation area
    SG_height, SG_width =SimulationGrid.shape[0], SimulationGrid.shape[1]
    SimulationGrid_List = np.ndarray.tolist(SimulationGrid)

    SGmax = SimulationGrid[np.isfinite(SimulationGrid)].max()
    SGmin = SimulationGrid[np.isfinite(SimulationGrid)].min()
    DS_Threshold = DS_DistanceThreshold_factor*(SGmax-SGmin)

    # specify hard data pattern
    DS_SearchingRadius = DS_PatternRadius
    Collection_y_List, Collection_x_List = Specify_ConditioningDataSequence_Spiral(DS_SearchingRadius)
    # Assign TI_SearchFraction 
    DS_Fraction = TI_SearchFraction
        
    ### specify the simulation path ###
    x = np.arange(0, SG_width)
    y = np.arange(0, SG_height)
    xx, yy = np.meshgrid(x, y)

        # index location of each pixel 
    Simulation_path_x_List = np.ndarray.tolist(xx.ravel().astype(int))
    Simulation_path_y_List = np.ndarray.tolist(yy.ravel().astype(int))
        # define simulation path by index numbers
    Simulation_path = np.arange(len(Simulation_path_x_List))
 
        # find locations with hard data observations & first simulate locations with hard data
    Simulation_path_hard = np.arange(len(Simulation_path_x_List))
    count=0
    for i in Simulation_path:
        if  np.any(np.isfinite(SimulationGrid[Simulation_path_y_List[i]-2:Simulation_path_y_List[i]+3, 
                                              Simulation_path_x_List[i]-2:Simulation_path_x_List[i]+3])):

            Simulation_path_hard[[count, i]] = Simulation_path_hard[[i, count]] 
            count+=1
        # randomize the path that is without harddata
    np.random.shuffle(Simulation_path_hard[count:]) 
    
    
    TI_list = np.ndarray.tolist(TI)

    for simulation_index in tq.tqdm(Simulation_path_hard):

        center_y = Simulation_path_y_List[simulation_index]
        center_x = Simulation_path_x_List[simulation_index]

        element = SimulationGrid_List[center_y][center_x]
        # visit an unknown point
        if np.isfinite(element):
            continue
        Conditioning_pattern_List, Conditioning_y_List, Conditioning_x_List = \
                                Extract_DSPattern_From_SimulationDomain(SimulationDomain_List = SimulationGrid_List,
                                                                        SG_height = SimulationGrid.shape[0],
                                                                        SG_width = SimulationGrid.shape[1],
                                                                        Center_y = center_y,
                                                                        Center_x = center_x,
                                                                        NeighborsAmount = DS_Neighbors,
                                                                        Collection_y_List = Collection_y_List,
                                                                        Collection_x_List = Collection_x_List)
        if(len(Conditioning_pattern_List)==0):
        # if the program does not find any hard data

            current_best_y  = np.random.randint(low=0,high=TI_height,size=1)[0]
            current_best_x  = np.random.randint(low=0,high=TI_width,size=1)[0]
            element = TI[current_best_y,current_best_x]

        else:
             # perform the exhaustive searching program to find a comparable pattern            
            current_best_y = 0
            current_best_x = 0
            current_best_TI_index = 0
            current_best_distance = 9999999.9


            # find the optimal value 
            training_y, training_x, training_distance = \
            SearchingProgram_DS_Exhaustive_v2(TrainingImage= TI,
                                              TI_height = TI.shape[0],
                                              TI_width = TI.shape[1],
                                              Conditioning_pattern = np.asarray(Conditioning_pattern_List),
                                              Conditioning_y = np.asarray(Conditioning_y_List),
                                              Conditioning_x = np.asarray(Conditioning_x_List),
                                              Distance_Threshold = DS_Threshold,
                                              Searching_Fraction = DS_Fraction)

            current_best_y = training_y
            current_best_x = training_x
        element = TI[current_best_y, current_best_x]
        SimulationGrid_List[center_y][center_x] = element
        
    return np.asarray(SimulationGrid_List)

def Specify_ConditioningDataSequence_Spiral(SearchingRadius):
    '''specify the sequence to collect hard data'''
    Collection_y_List = []
    Collection_x_List = []
    Collection_distance_List = []
    
    for y in range(-SearchingRadius,SearchingRadius+1):
        for x in range(-SearchingRadius,SearchingRadius+1):
            Collection_y_List.append(y)
            Collection_x_List.append(x)
            Collection_distance_List.append(y**2+x**2)
            
    Collection_y = np.array(Collection_y_List,dtype=int)
    Collection_x = np.array(Collection_x_List,dtype=int)
    Collection_distance = np.array(Collection_distance_List)
    
    Collection_y = Collection_y[ np.argsort(Collection_distance) ]
    Collection_x = Collection_x[ np.argsort(Collection_distance) ]
    
    Collection_y = np.delete(arr=Collection_y,obj=0)
    Collection_x = np.delete(arr=Collection_x,obj=0)
    
    
    return Collection_y.tolist(), Collection_x.tolist()

def Extract_DSPattern_From_SimulationDomain(SimulationDomain_List,
                                            SG_height, SG_width,
                                            Center_y, Center_x,
                                            NeighborsAmount,
                                            Collection_y_List,
                                            Collection_x_List):
    '''extract a DS pattern from simulation grid'''
    Conditioning_pattern_List = []
    Conditioning_y_List = []
    Conditioning_x_List = []
    
    for relative_y, relative_x in zip(Collection_y_List, Collection_x_List):
        location_y = Center_y + relative_y
        location_x = Center_x + relative_x
        
        if(location_y<0 or location_y>= SG_height or location_x<0 or location_x>=SG_width):
            continue
        
        element = SimulationDomain_List[location_y][location_x]
#         print(element)
        # only collect the point with hard data values
        if(np.isfinite(element)):
            Conditioning_pattern_List.append(element)
            Conditioning_y_List.append(relative_y)
            Conditioning_x_List.append(relative_x)
            
        if(len(Conditioning_pattern_List)==NeighborsAmount):
            return Conditioning_pattern_List, Conditioning_y_List, Conditioning_x_List
        
    return Conditioning_pattern_List, Conditioning_y_List, Conditioning_x_List

def SearchingProgram_DS_Exhaustive_v2(TrainingImage,
                                      TI_height, TI_width,
                                      Conditioning_pattern,
                                      Conditioning_y,
                                      Conditioning_x,
                                      Distance_Threshold, Searching_Fraction):
    
    '''Conduct a random sampling program to find a compatible pattern'''
    Searching_Num = int(TI_height * TI_width * Searching_Fraction)

    # specify the searching path
    Searching_path = np.arange(start=0,stop=TI_height*TI_width,step=1,dtype=int)
    np.random.shuffle(Searching_path)
    Searching_path = Searching_path[:Searching_Num]
    Searching_path_y_List = np.ndarray.tolist(Searching_path // TI_width)
    Searching_path_x_List = np.ndarray.tolist(Searching_path % TI_width)

    mean_Value = np.mean(TrainingImage)

    current_best_y = 0
    current_best_x = 0
    current_best_distance = 999999999.9

    Conditioning_n_sqrt = np.sqrt(len(Conditioning_pattern))

#     Conditioning_Num = len(Conditioning_pattern)

    # test a point in the training image
    for i in range(Searching_Num):
        # create a training pattern
        center_y, center_x = Searching_path_y_List[i], Searching_path_x_List[i]
        
        Training_pattern = np.repeat(mean_Value, len(Conditioning_y))

        y_patterns = center_y +Conditioning_y
        x_patterns = center_x +Conditioning_x
        
        patterns_in_TI = (x_patterns<TI_width) & (x_patterns>=0) & (y_patterns<TI_height) & (y_patterns>=0) 

        Training_pattern[patterns_in_TI] = TrainingImage[y_patterns[patterns_in_TI],x_patterns[patterns_in_TI]]


        # calculate distance between two patterns
        temp = Training_pattern-Conditioning_pattern
        pattern_distance = np.linalg.norm(temp)/Conditioning_n_sqrt
        
        if(pattern_distance <= Distance_Threshold):
            return center_y, center_x, pattern_distance
#             break
        elif(pattern_distance < current_best_distance):
            current_best_y = center_y
            current_best_x = center_x
            current_best_distance = pattern_distance

    return current_best_y, current_best_x, current_best_distance