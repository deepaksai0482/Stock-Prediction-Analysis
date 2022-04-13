from pyspark.sql import SparkSession;
spark = SparkSession.builder.master("local").appName("stocks").getOrCreate()

from pyspark.sql import SQLContext
from pyspark.sql.functions import col,lag
import pyspark.sql.functions as f
from pyspark.sql import Window
import yfinance as fi
from datetime import datetime
import pandas as pd
import plotly.express as px  
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)


s1 = input("Enter the first stock:")

s2 = input("Enter the second stock:")

fi_data1 = fi.download(s1,'2001-01-01',datetime.today().strftime('%Y-%m-%d'))

fi_data2 = fi.download(s2,'2001-01-01',datetime.today().strftime('%Y-%m-%d'))


raw_df_1 = spark.createDataFrame(data=fi_data1)
raw_df_1.printSchema();
raw_df_1.show()
raw_df_2 = spark.createDataFrame(data=fi_data2)
raw_df_2.printSchema();
raw_df_2.show()
raw_df_1.createOrReplaceTempView("stock1")
raw_df_2.createOrReplaceTempView("stock2")

df_1 = spark.sql(" select 'stock1' ,`Adj Close` as adjclose, ROW_NUMBER() OVER(order by 0) as row_num from stock1")
df_1.show()

w1 = Window.partitionBy('stock1').orderBy('row_num')
returns_df_1 = df_1.withColumn('prev_close',f.lag(col('adjclose')).over(w1))
returns_df_1.show()

returns_df_1.createOrReplaceTempView("stock11")
returns_df_1 = spark.sql("select row_num,stock1,adjclose,prev_close, ((adjclose)-(prev_close)) / (prev_close) as returns from stock11 ")
returns_df_1.show()

stock1_mean = returns_df_1.agg({'returns' : 'mean'})
stock1_variance = returns_df_1.agg({'returns' : 'variance'})
stock1_std_dev = returns_df_1.agg({'returns' : 'stddev'})

stock1_mean.show()
stock1_variance.show()
stock1_std_dev.show()

df_2 = spark.sql("select 'stock2',`Adj Close` as adjclose, ROW_NUMBER() OVER(order by 0) as row_num2 from stock2")
w2 = Window.partitionBy('stock2').orderBy('row_num2')
returns_df_2 = df_2.withColumn('prev_close',f.lag(col('adjclose')).over(w2))
returns_df_2.createOrReplaceTempView("stock21")
returns_df_2 = spark.sql("select row_num2,stock2,adjclose,prev_close, ((adjclose)-(prev_close)) / (prev_close) as returns from stock21 ")

stock2_mean = returns_df_2.agg({'returns' : 'mean'})
stock2_variance = returns_df_2.agg({'returns' : 'variance'})
stock2_std_dev = returns_df_2.agg({'returns' : 'stddev'})

stock2_mean.show()
stock2_variance.show()
stock2_std_dev.show()

stock1_mean_f = stock1_mean.head()[0]
stock2_mean_f = stock2_mean.head()[0]

stock1_std=stock1_std_dev.head()[0]
stock2_std=stock2_std_dev.head()[0]

print_value = 0;
if(stock1_std < stock2_std):
    print_value=print_value+1
    print("prefer the stock ",s1," than ",s2," Standard deviation of ",s1," is "," {:.4f}".format(stock1_std) ,"and Standard deviation of ",s2," is "," {:.4f}".format(stock2_std) )
    print(" In 68.3 % of time , For ",s1," the daily returns in between","{:.4f}".format(stock1_std-stock1_mean_f),"% and ","{:.4f}".format(stock1_std+stock1_mean_f),"%")
    print(" In 95.5 % of time , For ",s1," the daily returns in between","{:.4f}".format(stock1_std-stock1_mean_f-stock1_mean_f),"%  and ","{:.4f}".format(stock1_std+stock1_mean_f+stock1_mean_f),"%")
    print(" In 99.3 % of time , For ",s1," the daily returns in between","{:.4f}".format(stock1_std-stock1_mean_f-stock1_mean_f-stock1_mean_f),"%  and ","{:.4f}".format(stock1_std+stock1_mean_f+stock1_mean_f+stock1_mean_f),"%")
elif(stock1_std == stock2_std):
    print("Both stocks are same",s1," ",s2," Standard deviation of ",s1," is ", stock1_std ,"and Standard deviation of ",s2," is ", stock2_std)
    print("stock1 Prediction")
    print(" In 68.3 % of time , the daily returns in between","{:.4f}".format(stock1_std-stock1_mean_f),"% and ","{:.4f}".format(stock1_std+stock1_mean_f),"%")
    print(" In 95.5 % of time , the daily returns in between","{:.4f}".format(stock1_std-stock1_mean_f-stock1_mean_f),"%  and ","{:.4f}".format(stock1_std+stock1_mean_f+stock1_mean_f),"%")
    print(" In 99.3 % of time , the daily returns in between","{:.4f}".format(stock1_std-stock1_mean_f-stock1_mean_f-stock1_mean_f),"%  and ","{:.4f}".format(stock1_std+stock1_mean_f+stock1_mean_f+stock1_mean_f),"%")
    print("stock2 Prediction")
    print(" In 68.3 % of time , the daily returns in between","{:.4f}".format(stock2_std-stock2_mean_f),"% and ","{:.4f}".format(stock2_std+stock2_mean_f),"%")
    print(" In 95.5 % of time , the daily returns in between","{:.4f}".format(stock2_std-stock2_mean_f-stock2_mean_f),"% and ","{:.4f}".format(stock2_std+stock2_mean_f+stock2_mean_f),"%")
    print(" In 99.3 % of time , the daily returns in between","{:.4f}".format(stock2_std-stock2_mean_f-stock2_mean_f-stock2_mean_f),"% and ","{:.4f}".format(stock2_std+stock2_mean_f+stock2_mean_f+stock2_mean_f),"%")

else:
    print("prefer the stock ",s2," than ",s1," Standard deviation of ",s1, " is "," {:.4f}".format(stock1_std) ,"and Standard deviation of ",s2," is ","{:.4f}".format(stock2_std) )
    print(" In 68.3 % of time , For ",s2," the daily returns in between","{:.4f}".format(stock2_std-stock2_mean_f),"% and ","{:.4f}".format(stock2_std+stock2_mean_f),"%")
    print(" In 95.5 % of time , For ",s2," the daily returns in between","{:.4f}".format(stock2_std-stock2_mean_f-stock2_mean_f),"% and ","{:.4f}".format(stock2_std+stock2_mean_f+stock2_mean_f),"%")
    print(" In 99.3 % of time , For ",s2," the daily returns in between","{:.4f}".format(stock2_std-stock2_mean_f-stock2_mean_f-stock2_mean_f),"% and ","{:.4f}".format(stock2_std+stock2_mean_f+stock2_mean_f+stock2_mean_f),"%")

spark.stop()

# if(stock1_std < stock2_std):
#     list1 = [stock1_std-stock1_mean_f,stock1_std+stock1_mean_f]
#     list2 = [stock1_std-stock1_mean_f-stock1_mean_f,stock1_std+stock1_mean_f+stock1_mean_f]
#     list3 = [stock1_std-stock1_mean_f-stock1_mean_f-stock1_mean_f,stock1_std+stock1_mean_f+stock1_mean_f+stock1_mean_f]
# elif(stock1_std == stock2_std):
#     list1 = [stock1_std-stock1_mean_f,stock1_std+stock1_mean_f]
#     list2 = [stock1_std-stock1_mean_f-stock1_mean_f,stock1_std+stock1_mean_f+stock1_mean_f]
#     list3 = [stock1_std-stock1_mean_f-stock1_mean_f-stock1_mean_f,stock1_std+stock1_mean_f+stock1_mean_f+stock1_mean_f]
# else:
#     list4 = [stock2_std-stock2_mean_f,stock2_std+stock2_mean_f]
#     list5 = [stock2_std-stock2_mean_f-stock2_mean_f,stock2_std+stock2_mean_f+stock2_mean_f]
#     list6 = [stock2_std-stock2_mean_f-stock2_mean_f-stock2_mean_f,stock2_std+stock2_mean_f+stock2_mean_f+stock2_mean_f]

list0 = [stock1_std,stock2_std]
list1 = [stock1_std-stock1_mean_f,stock1_std+stock1_mean_f]
list2 = [stock1_std-stock1_mean_f-stock1_mean_f,stock1_std+stock1_mean_f+stock1_mean_f]
list3 = [stock1_std-stock1_mean_f-stock1_mean_f-stock1_mean_f,stock1_std+stock1_mean_f+stock1_mean_f+stock1_mean_f]
list4 = [stock2_std-stock2_mean_f,stock2_std+stock2_mean_f]
list5 = [stock2_std-stock2_mean_f-stock2_mean_f,stock2_std+stock2_mean_f+stock2_mean_f]
list6 = [stock2_std-stock2_mean_f-stock2_mean_f-stock2_mean_f,stock2_std+stock2_mean_f+stock2_mean_f+stock2_mean_f]

fig = go.Figure(data=[go.Scatter(x=list1, y=[0, 1, 2, 3])])
fig2 = go.Figure(data=[go.Scatter(x=list2, y=[0, 1, 2, 3])])
fig3 = go.Figure(data=[go.Scatter(x=list3, y=[0, 1, 2, 3])])
fig4 = go.Figure(data=[go.Scatter(x=list4, y=[0, 1, 2, 3])])
fig5 = go.Figure(data=[go.Scatter(x=list5, y=[0, 1, 2, 3])])
fig6 = go.Figure(data=[go.Scatter(x=list6, y=[0, 1, 2, 3])])

# App layout
if(stock1_std < stock2_std):
    app.layout = html.Div(children=[ 
                html.H1(children="Stock Prediction Analysis", style={'text-align': 'center'}),
                html.Div([ html.Div(children="Prefer the stock "+ s1 +" than "+s2+", the Standard deviation of \
                "+s1+" is "+ str(list0[0])[0:5] +" and Standard deviation of "+s2+" is "+\
                str(list0[1])[0:5], style={'text-align': 'center'})]),
               
                html.Div([
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-1''', style={'text-align': 'center'}),
                        html.Div(children="In 68.3 % of time , For " + s1 + " the daily returns \
                                 in between "+ str(list1[0])[0:5] + " and "+ str(list1[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig),
                    ], className='six columns'),
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-2''', style={'text-align': 'center'}),
                        html.Div(children="In 95.5 % of time , For " + s1 + " the daily returns \
                                 in between "+ str(list2[0])[0:5] + " and "+ str(list2[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig2),
                    ], className='six columns'),
        
                    ], className='row'),
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-3''', style={'text-align': 'center'}),
                        html.Div(children="In 99.3 % of time , For " + s1 + " the daily returns \
                                 in between "+ str(list3[0])[0:5] + " and "+ str(list3[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig3),
                    ], className='row'),
                html.Div([
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-4''', style={'text-align': 'center'}),
                        html.Div(children="In 68.3 % of time , For " + s2 + " the daily returns \
                                 in between "+ str(list4[0])[0:5] + " and "+ str(list4[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig4),
                    ], className='six columns'),
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-5''', style={'text-align': 'center'}),
                        html.Div(children="In 95.5 % of time , For " + s2 + " the daily returns \
                                 in between "+ str(list5[0])[0:5] + " and "+ str(list5[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig5),
                    ], className='six columns'),
        
                    ], className='row'),
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-6''', style={'text-align': 'center'}),
                        html.Div(children="In 99.3 % of time , For " + s2 + " the daily returns \
                                 in between "+ str(list6[0])[0:5] + " and "+ str(list6[1])[0:5], style={'text-align': 'center'}),
                    dcc.Graph(figure=fig6),
                    ], className='row'),
                    ])
else:
    app.layout = html.Div(children=[ 
                html.H1(children="Stock Prediction Analysis", style={'text-align': 'center'}),
                html.Div([ html.Div(children="prefer the stock "+s2+" than "+s1+" Standard deviation of \
                "+s2+" is "+ str(list0[1])[0:5] +" and Standard deviation of \
                "+s1+" is "+str(list0[0])[0:5] , style={'text-align': 'center'})]),
               
                html.Div([
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-1''', style={'text-align': 'center'}),
                        html.Div(children="In 68.3 % of time , For " + s1 + " the daily returns \
                                 in between "+ str(list1[0])[0:5] + " and "+ str(list1[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig),
                    ], className='six columns'),
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-2''', style={'text-align': 'center'}),
                        html.Div(children="In 95.5 % of time , For " + s1 + " the daily returns \
                                 in between "+ str(list2[0])[0:5] + " and "+ str(list2[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig2),
                    ], className='six columns'),
        
                    ], className='row'),
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-3''', style={'text-align': 'center'}),
                        html.Div(children="In 99.3 % of time , For " + s1 + " the daily returns \
                                 in between "+ str(list3[0])[0:5] + " and "+ str(list3[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig3),
                    ], className='row'),
                html.Div([
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-4''', style={'text-align': 'center'}),
                        html.Div(children="In 68.3 % of time , For " + s2 + " the daily returns \
                                 in between "+ str(list4[0])[0:5] + " and "+ str(list4[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig4),
                    ], className='six columns'),
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-5''', style={'text-align': 'center'}),
                        html.Div(children="In 95.5 % of time , For " + s2 + " the daily returns \
                                 in between "+ str(list5[0])[0:5] + " and "+ str(list5[1])[0:5], style={'text-align': 'center'}),
                        dcc.Graph(figure=fig5),
                    ], className='six columns'),
        
                    ], className='row'),
                    html.Div([
                        html.Br(),
                        html.Div(children='''Chart-6''', style={'text-align': 'center'}),
                        html.Div(children="In 99.3 % of time , For " + s2 + " the daily returns \
                                 in between "+ str(list6[0])[0:5] + " and "+ str(list6[1])[0:5], style={'text-align': 'center'}),
                    dcc.Graph(figure=fig6),
                    ], className='row'),
                    ])

if __name__ == '__main__':
    app.run_server(debug=False)                    
    
