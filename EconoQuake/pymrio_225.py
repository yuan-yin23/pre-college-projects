import pymrio
import warnings
import pandas as pd

eoraFolder = "/tmp/mrios/eora26"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    eoraLog = pymrio.download_eora26(
        storage_folder=eoraFolder,
        email="dsprints@gmu.edu",
        password="bJ5yN5848a5FJnR",
        years = ["2015"],
        overwrite_existing=True
    )
    eora = pymrio.parse_eora26(year = 2015, path = eoraFolder)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    eora.calc_all()


shockFactor = 2000  #Temporary variable given no existing shocks yet
shockCountry = "USA"
shockIndustry = "Agriculture"

originalOutput = eora.x.copy()
original_Y = eora.Y.copy()
L_matrix = eora.L.copy()

def update_shock_factor(value, country, industry):
    global shockFactor, shockCountry, shockIndustry
    shockFactor = value
    shockCountry = country
    shockIndustry = industry

def calculate_results():
    Y_shocked = eora.Y.copy()

    modifiers = ["Household final consumption P.3h"]#, "Non-profit institutions serving households P.3n",	"Government final consumption P.3g","Gross fixed capital formation P.51", "Changes in inventories P.52", "Acquisitions less disposals of valuables P.53"]

    for i in modifiers:
        try:
            Y_shocked.loc[(shockCountry, shockIndustry), (shockCountry, i)] -= shockFactor
        except KeyError:
            print(f"Invalid country/industry: ({shockCountry}, {shockIndustry})")
            continue

    #eoraNew = eora.copy()
    eoraTest = eora.copy()

    eoraTest.reset_all_to_coefficients()

    eoraTest.Y = Y_shocked

    # Y_total = Y_shocked.sum(axis=1)
    # x_new = L_matrix.dot(Y_total)

    # deltaX = x_new - originalOutput


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eoraTest.calc_all()
        leontief = eora.L
        Y_shocked_total = Y_shocked.sum(axis=1)
        x_new = leontief.dot(Y_shocked_total)
        deltaX = pd.Series()
      
        for i in x_new.keys():
            temp = pd.Series({i:(float  (x_new.loc[i] - originalOutput.loc[i]))})
            deltaX = pd.concat([deltaX, temp])


    return deltaX.to_dict();


'''returned = pd.DataFrame(deltaX)

returned.columns = ["simpleLeontief"]

returned.to_csv("/Users/graceim/Documents/Econ_research_Oughton/mrio_model/returned.csv")'''

'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    eoraTest.calc_all()
    outputChange = eoraTest.x - originalOutput
    usSectors = [i for i in eoraTest.x.index if i[0] == "USA"]
    #usChanges = outputChange.loc[usSectors, "indout"]
'''
'''
returned.index = outputChange.index
outputChange = outputChange.join(returned, how="left")


print(returned.head())
print(outputChange.head())


#originalOutput.to_csv("finalOutputMRIO.csv")
#this important
outputChange.to_csv("/Users/graceim/Documents/Econ_research_Oughton/mrio_model/globalOutputChangeMRIO_2.18.csv")
#usChanges.to_csv("nationalOutputChangeMRIO.csv")
'''