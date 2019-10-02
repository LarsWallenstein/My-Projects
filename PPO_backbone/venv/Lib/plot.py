import plotly.graph_objects as go
import plotly.express as px
from Search import *

from plotly.subplots import make_subplots

fig = make_subplots(rows=3, cols=1)

PATH = "C:/Users/corsa/Documents/Stocks/5 Min/ETFs/tvix.us.txt"
df = pd.read_csv(PATH, sep=',').round(5)
data = weis_wave(wt_lb(heikin_ashi(df)))

df = data.loc[200:300]

X = np.array(list(range(df.shape[0])))

fig.add_trace(
    go.Candlestick(x=X,
                   open=df['HaOpen'],
                   high=df['HaHigh'],
                   low=df['HaLow'],
                   close=df['HaClose']),
    row=1, col=1
)

fig.update_layout(xaxis_rangeslider_visible=False)

fig.add_trace(
go.Scatter(x=X, y=df['Wt2'],),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=X, y=df['Wt1']),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=X,y=df['Up']),
    row=3,col=1
)

fig.add_trace(
    go.Scatter(x=X,y=df['Down']),
    row=3,col=1
)
#fig.update_layout(xaxis_rangeslider_visible=True)
#fig.update_layout(height=600, width=800, title_text="Subplots")
fig.show()