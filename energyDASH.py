import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go

import joblib
import pandas as pd
import numpy as np

from PIL import Image























cleanDFcountries = joblib.load('cleanDFcountries.pkl')
modelosPrediccionesARIMA = joblib.load('modelosPrediccionesARIMA.pkl')
dfDescripcion = joblib.load('dfDescripcion.pkl')

prediccionNBEATSx = joblib.load('prediccionNBEATSx.pkl')
cleanDFcountriesMonthly = joblib.load('cleanDFScountriesMonthly.pkl')
metricasNBEATSx = joblib.load('metricasNBEATSx.pkl')

describeDF = joblib.load('describeDF.pkl')

columnsMonthly = ['Coal', 'Natural gas', 'Hydro', 'Nuclear', 'Oil', 'Solar', 'Wind']

describeDF = pd.read_csv('describeDF.csv')

df_allMetrics = joblib.load('df_allMetrics2.pkl')

columns = [
    'coal_electricity', 'coal_consumption',
    'gas_electricity', 'gas_consumption',
    'hydro_electricity', 'hydro_consumption',
    'nuclear_electricity', 'nuclear_consumption',
    'oil_electricity', 'oil_consumption',
    'solar_electricity', 'solar_consumption',
    'wind_electricity', 'wind_consumption'
]


columnasInteres = [
    'country',
    'year',
    'population',
    'gdp',
    'coal_electricity', 'coal_consumption',
    'gas_electricity', 'gas_consumption',
    'hydro_electricity', 'hydro_consumption',
    'nuclear_electricity', 'nuclear_consumption',
    'oil_electricity', 'oil_consumption',
    'solar_electricity', 'solar_consumption',
    'wind_electricity', 'wind_consumption'
]

dicCountries = {'Australia':0,'Austria':1,'Belgium':2,'Canada':3,'Chile':4,'Czech Republic':5,'Denmark':6,'Estonia':7,'Finland':8,'France':9,'Germany':10,'Greece':11,'Hungary':12,'IEA Total':13,'Iceland':14,'Ireland':15,'Italy':16,'Japan':17,'Korea':18,'Latvia':19,'Lithuania':20,'Luxembourg':21,'Mexico':22,'Netherlands':23,'New Zealand':24,'Norway':25,'OECD Americas':26,'OECD Asia Oceania':27,'OECD Europe':28,'OECD Total':29,'Poland':30,'Portugal':31,'Republic of Turkiye':32,'Slovak Republic':33,'Slovenia':34,'Spain':35,'Sweden':36,'Switzerland':37,'United Kingdom':38,'United States':39,'Colombia':40,'Argentina':41,'Brazil':42,'Bulgaria':43,'Croatia':44,'Cyprus':45,'India':46,'Malta':47,'North Macedonia':48,'Romania':49,'Serbia':50,'Costa Rica':51,}
dicEnergies = {'Coal':0,'Natural gas':1,'Hydro':2,'Nuclear':3,'Oil':4,'Solar':5,'Wind':6}

for df in cleanDFcountries:
    df['type'] = 'Historico'

for dataframe in cleanDFcountriesMonthly:
    dataframe['type'] = 'Historico'


for df in modelosPrediccionesARIMA:
    df['type'] = 'Predicho'

for dataframe in prediccionNBEATSx:
    for i in range(7):
        dataframe[i]['type'] = 'Predicho'

# Consolidar todos los DataFrames en uno solo
data = pd.concat(cleanDFcountries + modelosPrediccionesARIMA)

monthconcat = pd.concat(cleanDFcountriesMonthly)























# Crear la aplicación Dash
app = dash.Dash(__name__)
app.title = "Energías Dashboard"

## tabs
app.layout = html.Div(
    
    style={
        'backgroundColor': '#f0f0f0',  # Fondo general (gris claro)
        'padding': '20px',             # Espaciado alrededor
        'fontFamily': 'Arial, sans-serif'
    },
    children=[
        
        html.H1("Análisis y proyecciones energéticas para modelos predictivos de la generación y consumo de energias",
                style={'text-align': 'center'}),
        dcc.Tabs(
            id='tabs',
            value='eda',
            children=[
                dcc.Tab(label='EDA', value='eda'),
                dcc.Tab(label='BenchMarks', value='benchmarks'),
                dcc.Tab(label='Predicciones', value='predicciones'),
            ]
        ),
        html.Div(id='content')
    ]
)

@app.callback(
    Output('content', 'children'),
    Input('tabs', 'value')
)
def update_content(tab):























    ######## EDA
    if tab == 'eda':

        IMGcorrelationMatrix = Image.open("IMGcorrelationMatrix.png")
        IMGworldEnergyCons = Image.open("IMGworldEnergyCons.png")
        IMGworldEnergyProd = Image.open("IMGworldEnergyProd.png")
        IMGpie = Image.open("IMGpie.png")
        IMGworldEnergyProdMonthly = Image.open("IMGworldEnergyProdMonthly.png")

        return html.Div([

            html.H1("Análisis Exploratorio de Datos",
                    style={'text-align': 'center'}),

            html.H2("Análisis del dataSet Anual de la producción y consumo de energia",
                    style={'text-align': 'left'}),

            html.P( "todos los datos de las energias estan en Terawatt/hora (TWh)",
                style={'lineHeight': '1.5', 'textAlign': 'left', 'marginBottom': '20px',}),

            #HORIZONTAL
            html.Div([

                html.Img(src=IMGworldEnergyCons, 
                         style={'text-align': 'center','display': 'block','marginBottom': '20px','marginLeft': '20px','marginRight': '20px','maxWidth': '48%','height': 'auto',}),

                html.Img(src=IMGworldEnergyProd, 
                         style={'text-align': 'center','display': 'block','marginBottom': '20px','marginLeft': '20px','marginRight': '20px','maxWidth': '48%','height': 'auto',}),

            ],style={'display':'flex', 'justify-content': 'space-between' }),

            #HORIZONTAL
            html.Div([

                html.Img(src=IMGcorrelationMatrix,
                         style={'text-align': 'center','display': 'block','marginBottom': '20px','marginLeft': '20px','marginRight': '20px','maxWidth': '48%','height': 'auto',}),

                html.Img(src=IMGpie,
                         style={'text-align': 'center','display': 'block','marginBottom': '20px','marginLeft': '20px','marginRight': '20px','maxWidth': '48%','height': 'auto',}),

            ],style={'display':'flex', 'justify-content': 'space-between' }),

            html.H2("Análisis del dataSet Mensual de la producción y consumo de energia",
                    style={'text-align': 'left'}),

            html.Img(src=IMGworldEnergyProdMonthly, 
                     style={'text-align': 'center','display': 'block','marginBottom': '20px','marginLeft': '20px','marginRight': '20px','maxWidth': '98%','height': 'auto',}),

            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in describeDF.columns],  # Define las columnas
                data=describeDF.to_dict('records'),  # Convierte el DataFrame a un formato compatible
                style_table={'overflowX': 'auto', 'margin': '20px'},  # Para desplazamiento horizontal si es necesario
                style_cell={'textAlign': 'center', 'padding': '10px'},  # Ajustes de las celdas
                style_header={'backgroundColor': '#0074D9', 'color': 'white', 'fontWeight': 'bold'},  # Encabezado estilizado
            )


        ])






















    ######### benchmarks
    if tab == 'benchmarks':
        return html.Div([
            
            dcc.Tabs(
                id='benchmarks',
                value='knn',
                children=[
                    dcc.Tab(label='KNeighborsRegressor', value='knn'),
                    dcc.Tab(label='Ridge', value='ridge'),
                    dcc.Tab(label='Lasso', value='lasso'),
                    dcc.Tab(label='LinearRegression', value='linear'),
                    dcc.Tab(label='DecisionTreeRegressor', value='tree'),
                    dcc.Tab(label='RandomForestRegressor', value='forest'),
                    dcc.Tab(label='SVR', value='svr'),
                ]
            ),
            html.Div(id='benchmarks-content')  # Contenido dinámico de Benchmarks
        ])






















    ######## MODELO ORIGINAL
    if tab == 'modelooriginal':

        IMGnbeatsx = Image.open("IMGnbeatsx.png")

        return html.Div([

            html.H1("NBEATSx",
                    style={'text-align': 'center'}),

            html.H2("Neural basis expansion analysis with exogenous variables",
                    style={'text-align': 'center'}),

            #HORIZONTAL
            html.Div([

                html.Img(src=IMGnbeatsx,
                         style={'text-align': 'center','display': 'block','margin': '1 auto','maxWidth': '60%','height': 'auto',}),

                html.Div([

                    html.P("De la literatura se mira que NBEATSx es bueno para predecir series de tiempo que siguen un patron de temporadas",
                        style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),

                    dcc.Markdown('$$h_{s,b}=FCNN_{s,b}(y^{back}_{s,b-1},X_{b-1})$$',
                                  mathjax=True, style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),

                    dcc.Markdown('$$\Theta^{back} _{s,b} = LINEAR^{back}(h_{s,b})$$',
                                 mathjax=True, style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),

                    dcc.Markdown('$$\Theta^{for} _{s,b} = LINEAR^{for}(h_{s,b})$$',
                                 mathjax=True, style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),

                    html.P("block's basis vectors",
                        style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),

                    dcc.Markdown('$$\mathbf{V}_{s, b}^{back} \in \mathbb{R}^{L x N_s}$$ and $$\mathbf{V}_{s, b}^{for} \in \mathbb{R}^{H x N_s}$$',
                                 mathjax=True, style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),

                    dcc.Markdown('$$\hat{\mathbf{y}}^{back}_{s,b} = \mathbf{V}^{back}_{s,b}\mathbf{\Theta}^{back}_{s,b}$$ and $$\hat{\mathbf{y}}^{for}_{s,b} = \mathbf{V}^{for}_{s,b}\mathbf{\Theta}^{for}_{s,b}$$',
                                 mathjax=True, style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),    

                ])

            ],style={'display':'flex', 'justify-content': 'space-between' }),

            html.H2("MODEL INPUT",
                    style={'text-align': 'left','marginTop': '30px','marginLeft': '100px','marginRight': '100px',}),

            html.P("el modelo recibe la serie temporales Y com las exógenas X y lo que hace el modelo es descomponer la serie Y en los componentes TENDENCIA, ESTACIONALIDAD, EXÓGENOS los cuales muestran la evolucion, los patrones repetitivos y los factores que influyen",
                        style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),
            
            html.H2("STACKS & BLOCKS",
                    style={'text-align': 'left','marginTop': '30px','marginLeft': '100px','marginRight': '100px',}),

            html.P("Cada stack contiene múltiples bloques que procesan tanto datos históricos como variables exógenas mediante capas de redes neuronales. Cada bloque aprende patrones para generar predicciones, cuyos resultados se combinan y se envían al siguiente nivel para obtener una predicción global.",
                        style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),

            html.H2("FORECAST & BACKCAST",
                    style={'text-align': 'left','marginTop': '30px','marginLeft': '100px','marginRight': '100px',}),

            html.P("El pronóstico se genera en función de la salida global de todos los stacks. La proyección hacia atrás sirve para corregir posibles errores en la predicción, permitiendo que el modelo ajuste sus pronósticos según la información más reciente",
                        style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),

            html.H2("FC",
                    style={'text-align': 'left','marginTop': '30px','marginLeft': '100px','marginRight': '100px',}),

            html.P("Después de pasar por los bloques y stacks, el modelo usa una capa completamente conectada (FC) para refinar las predicciones. Aquí es donde el modelo hace los ajustes finales, utilizando parámetros específicos de cada bloque.",
                        style={'lineHeight': '1.5','textAlign': 'justify','marginBottom': '20px','marginLeft': '100px','marginRight': '100px',}),



            html.P( "referencia:",
                    style={'lineHeight': '1.5','textAlign': 'justify','marginTop': '100px','marginBottom': '1px'}),
            html.A("Olivares, K. G., Challu, C., Marcjasz, G., Weron, R., & Dubrawski, A. (2023). Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx. International Journal of Forecasting, 39(2), 884-900.", href="https://arxiv.org/pdf/2104.05522", target="_blank", style={'color': '#007BFF','lineHeight': '1.5','textAlign': 'justify','marginTop': '0px','marginBottom': '10px'}),
            

        ])






















    ######### PREDICCIONES
    if tab == 'predicciones':
        return html.Div([
            html.H1("NBEATSx ",
                    style={'text-align': 'center'}),

            #HORIZONTAL
            html.Div([
                dcc.Dropdown(
                    id='country-dropdown-predicciones',
                    options=[{'label': country, 'value': country} for country in monthconcat['country'].unique()],
                    value= monthconcat['country'].unique()[13],
                    placeholder="Selecciona un país",
                    style={'width': '50%', 'margin': '20px auto'}
                ),
                dcc.Dropdown(
                    id='energy-dropdown-predicciones',
                    options=[
                        {'label': 'Todo', 'value': 'all'},
                        {'label': 'Carbón', 'value': 'Coal'},
                        {'label': 'Gas', 'value': 'Natural gas'},
                        {'label': 'Hidro', 'value': 'Hydro'},
                        {'label': 'Nuclear', 'value': 'Nuclear'},
                        {'label': 'Petróleo', 'value': 'Oil'},
                        {'label': 'Solar', 'value': 'Solar'},
                        {'label': 'Eólica', 'value': 'Wind'},
                    ],
                    value='all',
                    placeholder="Selecciona una fuente de energía",
                    style={'width': '50%', 'margin': '20px auto'}
                ),
                
            ],style={'display':'flex', 'justify-content': 'space-between'}),

            dcc.Graph(id='energy-pred', style={'width': '98%', 'height': '800px', 'margin': '20px 20px'}),

            html.H2("Métrica del entrenamiento",
                    style={'text-align': 'center'}),


            
            dash_table.DataTable(
                id='metrics-table',
                columns=[{"name": col, "id": col} for col in ['NBEATSx', 'MAPE', 'RMSE', 'R²']],  # Columnas del DataFrame
                data=[],  # Datos iniciales vacíos
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'}
            )


            
        ])
    





















@app.callback(
    Output('benchmarks-content', 'children'),
    [Input('benchmarks', 'value')]
)
def render_benchmark_content(subtab):
    if subtab == 'knn':
        #CODE
        dataknn = df_allMetrics[df_allMetrics['Model'] == 'KNeighborsRegressor']
        dataknn['Parameters'] = dataknn['Parameters'].str[56:-2]
        return html.Div([

            html.H2("Resultados Gridsearch",
                style={'text-align': 'left'}),
                
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in dataknn.columns],  # Define las columnas
                data=dataknn.to_dict('records'),  # Convierte el DataFrame a un formato compatible
                style_table={'overflowX': 'auto', 'margin': '20px'},  # Para desplazamiento horizontal si es necesario
                style_cell={'textAlign': 'center', 'padding': '10px'},  # Ajustes de las celdas
                style_header={'backgroundColor': '#0074D9', 'color': 'white', 'fontWeight': 'bold'},  # Encabezado estilizado
            )

        ])
    
    elif subtab == 'ridge':
        #CODE
        dataknn = df_allMetrics[df_allMetrics['Model'] == 'Ridge']
        dataknn['Parameters'] = dataknn['Parameters'].str[37:-2]
        return html.Div([

            html.H2("Resultados Gridsearch",
                style={'text-align': 'left'}),
                
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in dataknn.columns],  # Define las columnas
                data=dataknn.to_dict('records'),  # Convierte el DataFrame a un formato compatible
                style_table={'overflowX': 'auto', 'margin': '20px'},  # Para desplazamiento horizontal si es necesario
                style_cell={'textAlign': 'center', 'padding': '10px'},  # Ajustes de las celdas
                style_header={'backgroundColor': '#0074D9', 'color': 'white', 'fontWeight': 'bold'},  # Encabezado estilizado
            )

        ])
    elif subtab == 'lasso':
        #CODE
        dataknn = df_allMetrics[df_allMetrics['Model'] == 'Lasso']
        dataknn['Parameters'] = dataknn['Parameters'].str[37:-2]
        return html.Div([

            html.H2("Resultados Gridsearch",
                style={'text-align': 'left'}),
                
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in dataknn.columns],  # Define las columnas
                data=dataknn.to_dict('records'),  # Convierte el DataFrame a un formato compatible
                style_table={'overflowX': 'auto', 'margin': '20px'},  # Para desplazamiento horizontal si es necesario
                style_cell={'textAlign': 'center', 'padding': '10px'},  # Ajustes de las celdas
                style_header={'backgroundColor': '#0074D9', 'color': 'white', 'fontWeight': 'bold'},  # Encabezado estilizado
            )

        ])
    elif subtab == 'linear':
        #CODE
        dataknn = df_allMetrics[df_allMetrics['Model'] == 'LinearRegression']
        return html.Div([

            html.H2("Resultados Gridsearch",
                style={'text-align': 'left'}),
                
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in dataknn.columns],  # Define las columnas
                data=dataknn.to_dict('records'),  # Convierte el DataFrame a un formato compatible
                style_table={'overflowX': 'auto', 'margin': '20px'},  # Para desplazamiento horizontal si es necesario
                style_cell={'textAlign': 'center', 'padding': '10px'},  # Ajustes de las celdas
                style_header={'backgroundColor': '#0074D9', 'color': 'white', 'fontWeight': 'bold'},  # Encabezado estilizado
            )

        ])
    elif subtab == 'tree':
        #CODE
        dataknn = df_allMetrics[df_allMetrics['Model'] == 'DecisionTreeRegressor']
        dataknn['Parameters'] = dataknn['Parameters'].str[55:-2]
        return html.Div([

            html.H2("Resultados Gridsearch",
                style={'text-align': 'left'}),
                
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in dataknn.columns],  # Define las columnas
                data=dataknn.to_dict('records'),  # Convierte el DataFrame a un formato compatible
                style_table={'overflowX': 'auto', 'margin': '20px'},  # Para desplazamiento horizontal si es necesario
                style_cell={'textAlign': 'center', 'padding': '10px'},  # Ajustes de las celdas
                style_header={'backgroundColor': '#0074D9', 'color': 'white', 'fontWeight': 'bold'},  # Encabezado estilizado
            )

        ])
    elif subtab == 'forest':
        #CODE
        dataknn = df_allMetrics[df_allMetrics['Model'] == 'RandomForestRegressor']
        dataknn['Parameters'] = dataknn['Parameters'].str[74:-2]
        return html.Div([

            html.H2("Resultados Gridsearch",
                style={'text-align': 'left'}),
                
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in dataknn.columns],  # Define las columnas
                data=dataknn.to_dict('records'),  # Convierte el DataFrame a un formato compatible
                style_table={'overflowX': 'auto', 'margin': '20px'},  # Para desplazamiento horizontal si es necesario
                style_cell={'textAlign': 'center', 'padding': '10px'},  # Ajustes de las celdas
                style_header={'backgroundColor': '#0074D9', 'color': 'white', 'fontWeight': 'bold'},  # Encabezado estilizado
            )

        ])
    elif subtab == 'svr':
        #CODE
        dataknn = df_allMetrics[df_allMetrics['Model'] == 'SVR']
        dataknn['Parameters'] = dataknn['Parameters'].str[48:-2]
        return html.Div([

            html.H2("Resultados Gridsearch",
                style={'text-align': 'left'}),
                
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in dataknn.columns],  # Define las columnas
                data=dataknn.to_dict('records'),  # Convierte el DataFrame a un formato compatible
                style_table={'overflowX': 'auto', 'margin': '20px'},  # Para desplazamiento horizontal si es necesario
                style_cell={'textAlign': 'center', 'padding': '10px'},  # Ajustes de las celdas
                style_header={'backgroundColor': '#0074D9', 'color': 'white', 'fontWeight': 'bold'},  # Encabezado estilizado
            )

        ])
    return html.Div("Selecciona un modelo para ver resultados.")





















# Callback para actualizar el gráfico de MODELOS
@app.callback(
    Output('population-trend', 'figure'),
    Output('gdp-trend', 'figure'),
    Output('energy-trend', 'figure'),
    Input('country-dropdown-modelos', 'value'),
    Input('energy-dropdown-modelos', 'value'),
)
def update_graph_modelos( selected_country, selected_energy):

    filtered_data = data[data['country'] == selected_country]

    colSel = []
    colSel = selected_energy
    if selected_energy == 'all':
        colSel = [
            'coal_electricity', 'coal_consumption',
            'gas_electricity', 'gas_consumption',
            'hydro_electricity', 'hydro_consumption',
            'nuclear_electricity', 'nuclear_consumption',
            'oil_electricity', 'oil_consumption',
            'solar_electricity', 'solar_consumption',
            'wind_electricity', 'wind_consumption'
        ]
    
    
    # Crear el gráfico de línea para Tab1
    population_fig = px.line(
        filtered_data,
        x='year',
        y=['population'],
        color='type',  # Diferenciar entre históricos y predicciones
        title=f'Tendencia de Poblacional - {selected_country}',
        labels={'value': 'Población', 'variable': 'Indicador'},
        #markers=True
    )

    gdp_fig = px.line(
        filtered_data,
        x='year',
        y=['gdp'],
        color='type',  # Diferenciar entre históricos y predicciones
        title=f'Tendencia de GDP - {selected_country}',
        labels={'value': 'GDP', 'variable': 'Indicador'},
        #markers=True
    )

    energy_fig = px.line(
        filtered_data,
        x='year',
        y=colSel,
        color='type',
        title=f'Tendencia Energética - {selected_country} {selected_energy}',
        labels={'value': 'Consumo Energético (TWh)', 'variable': 'Fuente Energética'},
        #markers=True
    )

    if selected_energy == 'all':
        max_y = filtered_data[columns].max().max()
        min_y = filtered_data[columns].min().min()
    else:
        max_y = filtered_data[selected_energy].max()
        min_y = filtered_data[selected_energy].min()

    # Asegurar que el rango tenga márgenes adecuados
    energy_fig.update_yaxes(range=[
        min(-0.1, min_y * 1.1),  # Margen inferior, asegurando que no sea menor a 0
        max(0.1, max_y * 1.1)  # Margen superior, asegurando mínimo de 0.1
    ])

    
    return population_fig, gdp_fig, energy_fig





















# Callback para actualizar el gráfico de PREDICCIONES
@app.callback(
    Output('energy-pred', 'figure'),
     Output('metrics-table', 'data'),
    Input('country-dropdown-predicciones', 'value'),
    Input('energy-dropdown-predicciones', 'value'),
)
def update_graph_predicciones(selected_country, selected_energy):

    # Filtrar los datos según la categoría seleccionada en Tab2
    filtered_data = monthconcat[monthconcat['country'] == selected_country]

    colSel = []
    colSel = selected_energy
    if selected_energy == 'all':
        colSel = ['Coal', 'Natural gas', 'Hydro', 'Nuclear', 'Oil', 'Solar', 'Wind']

    
    # Crear el gráfico de dispersión para Tab2
    fig_pred = px.line(
        filtered_data,
        x="year",
        y=colSel,
        color="type",
        title=f"Produccion de energia {selected_country} - {selected_energy}",
        #markers=True
    )

    if selected_energy == 'all':
        max_y = filtered_data[columnsMonthly].max().max()
        min_y = filtered_data[columnsMonthly].min().min()
    else:
        max_y = filtered_data[selected_energy].max()
        min_y = filtered_data[selected_energy].min()

    # Asegurar que el rango tenga márgenes adecuados
    fig_pred.update_yaxes(range=[
        min(-0.1, min_y * 1.1),  # Margen inferior, asegurando que no sea menor a 0
        max(0.1, max_y * 1.1)  # Margen superior, asegurando mínimo de 0.1
    ])


    if selected_energy != 'all':
        c = dicCountries[selected_country]
        e = dicEnergies[selected_energy]

        df_p = prediccionNBEATSx[c][e]
        df_m = metricasNBEATSx[c][e]

        fig_pred.add_trace(go.Scatter(
            x=df_p['ds']+ pd.DateOffset(years=1),
            y=df_p['NBEATSx-lo-90'],
            fill=None,  # No llenar abajo
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),  # Línea invisible
            showlegend=False
        ))

        fig_pred.add_trace(go.Scatter(
            x=df_p['ds']+ pd.DateOffset(years=1),
            y=df_p['NBEATSx-hi-90'],
            fill='tonexty',  # Rellenar hacia la línea siguiente
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),  # Línea invisible
            fillcolor='rgba(255, 0, 0, 0.2)',  # Color del área de relleno (semi-transparente)
            showlegend=False
        ))

        fig_pred.add_trace(go.Scatter(
            x=df_p['ds']+ pd.DateOffset(years=1),
            y=df_p['NBEATSx-median'],
            mode='lines',
            name='Predicho',
            line=dict(color='red'),
            showlegend=True
        ))

        table_data = df_m.to_dict('records')
    else:
        table_data = []
    
    return fig_pred, table_data





















# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
