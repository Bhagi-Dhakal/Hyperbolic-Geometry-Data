import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
   
from scipy.stats import norm


import datetime
import os
import ast

import math
import statistics


"""
Used to Gather Data from SnapyPy

import json  
Manifold_counter = 0
for M in OrientableClosedCensus[Manifold_counter:len(OrientableClosedCensus)]:

    try: 
        file_path = '/data/RealData/' + str(Manifold_counter + 1) + '.json'
        data = M.length_spectrum(6)

        data_dict = {}

        data_dict['Manifold_index'] = Manifold_counter
        data_dict['Name'] = M.name()
        data_dict['Betti_Number']=  M.homology().betti_number()
        geodesics = []

        for i in range(len(data)):
            geo_x = []
            geo_x.append(int(data[i].multiplicity))
            
            geo_x.append(float(data[i].length.real()))
            geo_x.append(float(data[i].length.imag()))
            geodesics.append(geo_x)

        data_dict['Geodesics_MLH']=  geodesics

        with open(file_path , 'w') as file:
            json.dump(data_dict, file, indent=4)

        file.close()

    except RuntimeError as e:
        print(f"Error at Manifold {Manifold_counter}: {e}")
        print(f"Skipping Manifold {Manifold_counter} and continuing.")

    finally:
        Manifold_counter += 1 



Problems with the following Manifolds: Actucal index in the list
nodata_ind = [ 647, 1900, 1990, 2124, 2254, 2311, 2499, 2511, 2557,  2613, 2787, 2899, 
               3043, 3613, 3793, 4086, 4234, 4392, 4473, 4789, 4860, 5452, 5493, 5641, 
               5718, 6026, 6282, 6439, 6850, 6986, 6990, 7054, 7157, 7173, 7255, 7282, 
               7504, 7529, 7546, 7596, 7605, 7787, 8046, 8074, 8235, 8448, 8644, 8707, 
               8882, 9137, 9144, 9240, 9267, 9294, 9686, 9867, 10261, 10458, 10611] # 59
"""


def count_files(path):
    x = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    return x + 59


nodata_ind = [ 647, 1900, 1990, 2124, 2254, 2311, 2499, 2511, 2557,  2613, 2787, 2899, 
               3043, 3613, 3793, 4086, 4234, 4392, 4473, 4789, 4860, 5452, 5493, 5641, 
               5718, 6026, 6282, 6439, 6850, 6986, 6990, 7054, 7157, 7173, 7255, 7282, 
               7504, 7529, 7546, 7596, 7605, 7787, 8046, 8074, 8235, 8448, 8644, 8707, 
               8882, 9137, 9144, 9240, 9267, 9294, 9686, 9867, 10261, 10458, 10611] # 59

# Charts 
plt.rcParams['text.usetex'] = True
def create_bar_chart(data, totals = False, many_bins = False, name = 'Name', betti_number = -1):

    
    output_directory = "/Results/GraphsAndCharts"

    if betti_number >= 0:
        values = []

        range1 = []
        range2 = []
        range3 = []
        range4 = []
        range5 = []

        if betti_number == 0:
            for i, item in enumerate(data[0]):
                if i in data[1][0] or i in data[1][1]:
                    continue
                else:
                    values.append(item)

        if betti_number == 1:
            for iele in data[1][0]:          
                values.append(data[0][iele])  

        if betti_number == 2:
            for iele in data[1][1]:
                values.append(data[0][iele]) 

        
        for element in values:
            if 0 <= element < 20:
                range1.append(element)
            elif 20 <= element < 40:
                range2.append(element)
            elif 40 <= element < 60:
                range3.append(element)
            elif 60 <= element < 80:
                range4.append(element)
            elif 80 <= element <= 100:
                range5.append(element)

        values = [range1, range2, range3, range4, range5]
        d = []

        for i in range(len(values)):
            d.append(len(values[i]))

        labels = ['0 - 20 %', '20 - 40 %', '40 - 60 %','60 - 80 %','80 - 100 %']

        plt.gcf().set_facecolor('lightgray')  
        plt.gca().set_facecolor('lightgray') 

        bars = plt.bar(labels, d, color = [(0.4, 0.6, 1.0),(1.0, 0.6, 0.2),(0.4, 1.0, 0.4), (1.0, 0.4, 0.4), (1.0, 1.0, 0.4)])
        for bar, value in zip(bars, d):
            x = bar.get_x() + bar.get_width() / 2  
            y = max(d) / 2                      
            plt.text(
                x, y,               
                str(value),         
                ha='center',        
                va='center',        
                color='black',      
                fontsize=12       
            )

        output_path = os.path.join(output_directory, f"LeftWinningPercentBN{betti_number}.png")

        plt.xlabel('')
        plt.ylabel("Counts")
        plt.title(f"Left Winning % for Betti Number:{betti_number}")

        plt.savefig(output_path, dpi=300)
        plt.close()
    
    if totals:
        values = []
        left_count = len(data[0])
        right_count = len(data[1])

        labels = ['Left Totals', 'Right Totals']
        values.append(left_count)
        values.append(right_count)

        plt.gcf().set_facecolor('lightgray')  
        plt.gca().set_facecolor('lightgray') 

        bars = plt.bar(labels, values, color=[(0.4, 0.4, 1), (1.0, 0.4, 0.4)])
        for bar, value in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2  
            y = max(values) / 2                
            plt.text(
                x, y,               
                str(value),         
                ha='center',        
                va='center',        
                color='black',      
                fontsize=12         
            )

        output_path = os.path.join(output_directory, f"LeftRightWinsTotal.png")

        plt.xlabel('')
        plt.ylabel("Counts")
        plt.title("Left vs Right Wins across all Manifolds")

        plt.savefig(output_path, dpi=300)
        plt.close()
    
    if many_bins:
        values = []
        labels = ['0 - 20 %', '20 - 40 %', '40 - 60 %','60 - 80 %','80 - 100 %']

        for i in range(len(data)):
            values.append(len(data[i]))

        plt.gcf().set_facecolor('lightgray')  
        plt.gca().set_facecolor('lightgray') 

        bars = plt.bar(labels, values, 
                        color = [(0.4, 0.6, 1.0),(1.0, 0.6, 0.2),(0.4, 1.0, 0.4), (1.0, 0.4, 0.4), (1.0, 1.0, 0.4)]
                    )


        for bar, value in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2  
            y = max(values) / 2      
            plt.text(
                x, y,               
                str(value),         
                ha='center',        
                va='center',        
                color='black',      
                fontsize=12         
            )


        output_path = os.path.join(output_directory, "LeftWinningPercentTotals.png")

        plt.xlabel('')
        plt.ylabel("Counts")
        plt.title(f"Left Winning % across all manifolds")

        plt.savefig(output_path, dpi=300)
        plt.close()


def create_histogram(data, save_name = 'Save', graph_name = 'graph', betti_number= -1, ):

    if betti_number == -1:
        plt.hist(data, bins=15, color=(.4,.4,1), edgecolor='black', alpha=0.7, density=True)
        
        mean, std = statistics.mean(data), statistics.stdev(data)
        x = np.linspace(min(data), max(data), 100)
        pdf = norm.pdf(x, mean, std)

        plt.gcf().set_facecolor('lightgray')  
        plt.gca().set_facecolor('lightgray') 
        
        plt.plot(x, pdf, color=(1, 0.4, 0.4), linewidth=2, label=f'Normal Distribution\n$\mu={mean:.2f}, \sigma={std:.2f}$')    


        plt.title(graph_name)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        output_directory = "/Results/GraphsAndCharts"  
        output_path = os.path.join(output_directory, f"{save_name}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

    if betti_number >= 0:

        processed_data = []

        if betti_number == 0:
            for i, item in enumerate(data[0]):
                if i in data[1][0] or i in data[1][1]:
                    continue
                else:
                    processed_data.append(item)

        if betti_number == 1:
            for iele in data[1][0]:        
                processed_data.append(data[0][iele])  

        if betti_number == 2:
            for iele in data[1][1]:
                processed_data.append(data[0][iele]) 


        plt.hist(processed_data, bins=15, color=(.4,.4,1), edgecolor='black', alpha=0.7, density=True)
        
        mean = statistics.mean(processed_data)

        if betti_number != 2:
            std = statistics.stdev(processed_data)
        else:
            std = 0

        x = np.linspace(min(processed_data), max(processed_data), 100)
        pdf = norm.pdf(x, mean, std)

        plt.gcf().set_facecolor('lightgray')  
        plt.gca().set_facecolor('lightgray') 
        
        plt.plot(x, pdf, color=(1, 0.4, 0.4), linewidth=2, label=f'Normal Distribution\n$\mu={mean:.2f}, \sigma={std:.2f}$')    

        plt.title(graph_name)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        output_directory = "Results/GraphsAndCharts"  
        output_path = os.path.join(output_directory, f"{save_name}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


### Main Analysis Function
def run_i_analysis(data_path, save_path):

    with open(data_path, "r") as file:
        data = ast.literal_eval(file.read())
    file.close()
    geo = data['Geodesics_MLH']

    left = 0
    right = 0
    left_winning = 0
    right_winning = 0  
    tie = 0
    left_p_winning = -1

    N_RightMinusLeft = []

    Sum_CosHolonomy = 0
    N_upto_Sum_CosHolonomy = []

    Sum_LengthTimesCosHolonomy = 0
    N_upto_Sum_LengthTimesCosHolonomy = []

    for geodesic in geo: 
        holonomy_value = float(geodesic[2])
        Length_value = float(geodesic[1])

        for i in range(int(geodesic[0])):

            if (math.pi / 2) < holonomy_value < (3 * math.pi / 2) or (-3 * math.pi / 2) < holonomy_value < (-math.pi / 2):
                left += 2
            else:
                right += 2
            N_RightMinusLeft.append((right - left) / math.e ** Length_value )

            if (right - left) < 0:
                left_winning += 1
            elif (right - left) > 0:
                right_winning += 1
            else:
                tie += 1

        Sum_CosHolonomy += 2 * int(geodesic[0]) * math.cos(holonomy_value)
        N_upto_Sum_CosHolonomy.append(Sum_CosHolonomy / math.e ** Length_value)

        Sum_LengthTimesCosHolonomy += 2 * int(geodesic[0]) * math.cos(holonomy_value) * Length_value
        N_upto_Sum_LengthTimesCosHolonomy.append(Sum_LengthTimesCosHolonomy / ((math.e) ** Length_value))

    left_p_winning = ((left_winning / (left_winning + right_winning + tie)) * 100)

    Mean_N_RightMinusLeft = statistics.mean(N_RightMinusLeft) 

    Mean_N_upto_Sum_CosHolonomy = statistics.mean(N_upto_Sum_CosHolonomy) 

    Mean_N_upto_Sum_LengthTimesCosHolonomy = statistics.mean(N_upto_Sum_LengthTimesCosHolonomy)

    results =  {
        'Name' : data['Name'],
        'Betti_Number' : data['Betti_Number'],

        'left': left,
        'right': right,

        'left_winning': left_winning,
        'right_winning': right_winning,
        'tie': tie,
        'left_p_winning' : left_p_winning,

        'N_RightMinusLeft' : N_RightMinusLeft, 
        'Mean_N_RightMinusLeft': Mean_N_RightMinusLeft,

        'Sum_CosHolonomy': Sum_CosHolonomy,
        'N_upto_Sum_CosHolonomy' : N_upto_Sum_CosHolonomy,
        'Mean_N_upto_Sum_CosHolonomy': Mean_N_upto_Sum_CosHolonomy,

        'Sum_LengthTimesCosHolonomy': Sum_LengthTimesCosHolonomy,
        'N_upto_Sum_LengthTimesCosHolonomy' : N_upto_Sum_LengthTimesCosHolonomy,
        'Mean_N_upto_Sum_LengthTimesCosHolonomy': Mean_N_upto_Sum_LengthTimesCosHolonomy
    }


    with open(save_path, "w") as file:
        file.write(repr(results))

    file.close()


def analyze_allFiles(data_path, save_folder):


    for i in range(count_files(data_path)):
    
        if i in nodata_ind: 
            continue

        data_path_i = os.path.join(data_path, f"{i + 1}.json")
        save_path_i = os.path.join(save_folder, f"M{i + 1}.txt")

        run_i_analysis(data_path_i, save_path_i)
        print(f"Done looking at Manifold:{i+1}")

##############################################################################################################################
analyze_allFiles("/data/RealData", "Results/Manifold_ind")

##############
data_path = "/data/RealData"


def multi_analysis(data_path_dir, save_dir):
    
    left_totals = [] 
    right_totals = []

    left_p_winning_overall = [] 
    range1 = []
    range2 = []
    range3 = []
    range4 = []
    range5 = []

    Mean_N_RightMinusLeft = [] 

    manifold_index_betti1 = []
    manifold_index_betti2 = [] 

    Mean_N_upto_Sum_CosHolonomy = [] 

    Mean_N_upto_Sum_LengthTimesCosHolonomy = [] 

    for i in range(count_files(data_path_dir)):

    
        if i in nodata_ind: 
            continue

        file_path = os.path.join(data_path_dir, f'M{i+1}.txt')

        with open(file_path, 'r') as f:
            data = ast.literal_eval(f.read().strip())

        left_p_winning_overall.append(data['left_p_winning'])

        if data['Betti_Number'] == 1:
            manifold_index_betti1.append(i)

        if data['Betti_Number'] == 2:
            manifold_index_betti2.append(i)
        

        if(data['left'] > data['right']):
            left_totals.append(data['left'])
        else:
            right_totals.append(data['right'])

       

        if 0 <= data['left_p_winning'] < 20:
            range1.append(data['left_p_winning'])
        elif 20 <= data['left_p_winning'] < 40:
            range2.append(data['left_p_winning'])
        elif 40 <= data['left_p_winning'] < 60:
            range3.append(data['left_p_winning'])
        elif 60 <= data['left_p_winning'] < 80:
            range4.append(data['left_p_winning'])
        elif 80 <= data['left_p_winning'] <= 100:
            range5.append(data['left_p_winning'])

        

        Mean_N_RightMinusLeft.append(data['Mean_N_RightMinusLeft'])

        Mean_N_upto_Sum_CosHolonomy.append(data['Mean_N_upto_Sum_CosHolonomy'])

        Mean_N_upto_Sum_LengthTimesCosHolonomy.append(data['Mean_N_upto_Sum_LengthTimesCosHolonomy'])

        f.close()
        print(f'Finished adding file:{i+1} info')

    print(f"Betti Number1:{manifold_index_betti1}")
    print(f"Betti Number2:{manifold_index_betti2}")


    left_p_winning = [range1, range2, range3, range4, range5]

    overall = {}
    overall['left_totals'] = left_totals
    overall['right_totals'] = right_totals

    overall['left_p_winning'] = left_p_winning
    overall['left_p_winning_overall'] = left_p_winning_overall
    overall['Mean_N_RightMinusLeft'] = Mean_N_RightMinusLeft
    overall['manifold_index_betti1'] = manifold_index_betti1
    overall['manifold_index_betti2'] = manifold_index_betti2
    overall['Mean_N_upto_Sum_CosHolonomy'] = Mean_N_upto_Sum_CosHolonomy
    overall['Mean_N_upto_Sum_LengthTimesCosHolonomy'] = Mean_N_upto_Sum_LengthTimesCosHolonomy


    save_path = os.path.join(save_dir, 'Overall.txt')
    with open(save_path, "w") as file:
        file.write(repr(overall))


    # creating graphs:
    create_bar_chart([left_totals, right_totals], totals = True, name='Total_wins_per_side') 

    create_bar_chart(left_p_winning, many_bins = True, name = 'Left_winning_%') 
    create_bar_chart([left_p_winning_overall, [manifold_index_betti1, manifold_index_betti2]], name = 'Left_winning_%_B1', betti_number = 1)
    create_bar_chart([left_p_winning_overall, [manifold_index_betti1, manifold_index_betti2]], name = 'Left_winning_%_B2', betti_number = 2)
    create_bar_chart([left_p_winning_overall, [manifold_index_betti1, manifold_index_betti2]], name = 'Left_winning_%_B0', betti_number = 0)

    create_histogram(Mean_N_RightMinusLeft, save_name = 'RightMinusLeftOverall', graph_name = r'Average value of $RL_{\Gamma}(6)$')
    create_histogram([Mean_N_RightMinusLeft, [manifold_index_betti1, manifold_index_betti2]], save_name = 'RightMinusLeftBN0', graph_name = r'Average value of $RL_{\Gamma}(6)$ For Betti Number:0', betti_number = 0)
    create_histogram([Mean_N_RightMinusLeft, [manifold_index_betti1, manifold_index_betti2]], save_name = 'RightMinusLeftBN1', graph_name = r'Average value of $RL_{\Gamma}(6)$ For Betti Number:1', betti_number = 1)
    create_histogram([Mean_N_RightMinusLeft, [manifold_index_betti1, manifold_index_betti2]], save_name = 'RightMinusLeftBN2', graph_name = r'Average value of $RL_{\Gamma}(6)$ For Betti Number:2', betti_number = 2)

    create_histogram(Mean_N_upto_Sum_CosHolonomy, save_name = "CosHolonomyOverall", graph_name = r'Average value of $K_{\Gamma}(6)$')
    create_histogram([Mean_N_upto_Sum_CosHolonomy, [manifold_index_betti1, manifold_index_betti2]], save_name = "CosHolonomyBN0", graph_name = r'Average value of $K_{\Gamma}(6)$ For Betti Number:0', betti_number = 0)
    create_histogram([Mean_N_upto_Sum_CosHolonomy, [manifold_index_betti1, manifold_index_betti2]], save_name = "CosHolonomyBN1", graph_name = r'Average value of $K_{\Gamma}(6)$ For Betti Number:1', betti_number = 1)
    create_histogram([Mean_N_upto_Sum_CosHolonomy, [manifold_index_betti1, manifold_index_betti2]], save_name = "CosHolonomyBN2", graph_name = r'Average value of $K_{\Gamma}(6)$ For Betti Number:2', betti_number = 2)

    create_histogram(Mean_N_upto_Sum_LengthTimesCosHolonomy, save_name ="LenghtTimesCosHolonomyOverall", graph_name = r'Average value of $L_{\Gamma}(6)$')
    create_histogram([Mean_N_upto_Sum_LengthTimesCosHolonomy, [manifold_index_betti1, manifold_index_betti2]], save_name ="LengthTimesCosHolonomyBN0", graph_name = r'Average value of $L_{\Gamma}(6)$ For Betti Number:0', betti_number = 0)
    create_histogram([Mean_N_upto_Sum_LengthTimesCosHolonomy, [manifold_index_betti1, manifold_index_betti2]], save_name ="LengthTimesCosHolonomyBN1", graph_name = r'Average value of $L_{\Gamma}(6)$ For Betti Number:1', betti_number = 1)
    create_histogram([Mean_N_upto_Sum_LengthTimesCosHolonomy, [manifold_index_betti1, manifold_index_betti2]], save_name ="LengthTimesCosHolonomyBN2", graph_name = r'Average value of $L_{\Gamma}(6)$ For Betti Number:2', betti_number = 2)






    

multi_analysis("/Results/Manifold_ind", "/Results/Manifold_overall")



