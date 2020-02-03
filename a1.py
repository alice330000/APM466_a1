import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
import scipy.optimize as optimize
from scipy.interpolate import interp1d
import numpy as np
from numpy import linalg as LA


# read the data file.
xls = pd.ExcelFile('/Users/zhangwanying/Desktop/data for APM466 AS1.xlsx')
ds1 = pd.read_excel(xls, '1.2')
ds2 = pd.read_excel(xls, '1.3')
ds3 = pd.read_excel(xls, '1.6')
ds4 = pd.read_excel(xls, '1.7')
ds5 = pd.read_excel(xls, '1.8')
ds6 = pd.read_excel(xls, '1.9')
ds7 = pd.read_excel(xls, '1.10')
ds8 = pd.read_excel(xls, '1.13')
ds9 = pd.read_excel(xls, '1.14')
ds10 = pd.read_excel(xls, '1.15')
ds = [ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10]
labels = ['Jan 2','Jan 3','Jan 6','Jan 7','Jan 8', 'Jan 9','Jan 10','Jan 13','Jan 14','Jan 15']


def time_to_maturity(data):
    curr_date = list(data.columns.values)[0]
    temp = []
    for md in data['maturity date']:
        temp.append((md - curr_date).days)
    data['time to maturity'] = temp

def AI(data):
    coupon_period = 182
    temp = []
    for index, row in data.iterrows():
        rest_days = row["time to maturity"] % coupon_period
        temp.append((coupon_period - rest_days) * row["coupon"] * 100 / 365)
    data["accrued interest"] = temp

# convert clean price to dirty price
def DP(data):
    temp = []
    for index, row in data.iterrows():
        temp.append (row["close price"] + row["accrued interest"])
    data["dirty price"] = temp

# Evaluate the yield to maturity
def evaluate_yield(data):
    yield_y_axis, time_x_axis = [], []

    for index, row in data.iterrows ():
        coupon_period = 182
        rest_days = row["time to maturity"]
        initial_days = (rest_days % coupon_period) / coupon_period

        coupon_num = int(rest_days / coupon_period)
        time = np.asarray([initial_days + n for n in range(0, coupon_num + 1)])

        coupon = row["coupon"] * 100 / 2
        payment = np.asarray([coupon] * coupon_num + [coupon + 100])
        time_x_axis.append (rest_days / 365)

        solve_ytm = lambda y: np.dot(payment, (1 + y / 2) ** (-time)) - row["dirty price"]

        ytm = optimize.fsolve(solve_ytm, 0.03)
        yield_y_axis.append(ytm)
    data["y"], data["t"] =  yield_y_axis, time_x_axis

# Plot yield curve with initial data
def plot_ytm(selected_bonds):
    plt.xlabel('time to maturity')
    plt.ylabel('yield to maturity')
    plt.title('5-year YTM curve')
    for index in range(0,10):
        plt.plot(selected_bonds[index]["t"], selected_bonds[index]["y"], label = labels[index])
        plt.legend (loc = 'upper left', bbox_to_anchor = (0.77, 0.95))
    plt.show()
plot_ytm(ds)

# add new useful data to selected bonds
def add_new_data(selected_bonds):
    for data in selected_bonds:
        time_to_maturity(data)
        AI(data)
        DP(data)
        evaluate_yield(data)
add_new_data(ds)

# Interpolation method
def ip_ytm(x_axis, y_axis):
    time_lag = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    inter = interp1d(x_axis, y_axis, bounds_error=False)
    temp = []
    for time in time_lag:
        outcome = float(inter(time))
        temp.append(outcome)
    return np.asarray(time_lag), np.asarray(temp)

# Plot the interpolated yield curve
def plot_ip_ytm(selected_bonds):
    plt.xlabel('time to maturity')
    plt.ylabel('yield to maturity')
    plt.title('interpolated 5-year yield curve')
    for index in range(0,10):
        inter_res = ip_ytm(selected_bonds[index]["t"], selected_bonds[index]["y"])
        plt.plot(inter_res[0], inter_res[1].squeeze(), label = labels[index])
    plt.legend(loc = 'upper right', bbox_to_anchor = (0.99, 0.99))
    plt.show()
plot_ip_ytm(ds)


# Calculate the spot rate
def evaluate_spot(data):
    spots = []
    for index, row in data.iterrows():
        dirty_price = data["dirty price"]
        semi_coupons = data["coupon"] * 100 / 2
        time = data['time to maturity'] * 2
        initial_days = 4/6

        if time == 0:
            r_1 = -np.log(dirty_price / (semi_coupons + 100)) / data["t"]
            spots.append(r_1 * 2)
        else:
            cf = 0
            # print(type(bonds["plot x"][:i]))
            for i in range(1, int(time)+1):
                cf += semi_coupons * np.exp(-(spots[i-1]/2)*(1-initial_days))
            spot_func = lambda y: np.dot(cf,
                        np.exp(-(np.multiply(spots[index], data["t"][index])))) + cf * np.exp(-y * data["t"]) - dirty_price
            rate = optimize.fsolve(spot_func, 0.03)
            spots.append(rate)
        for spot in spots:
            print spot




# Plot the spot curve with initial curve
def plot_spot(data):
    plt.xlabel('time to maturity')
    plt.ylabel('spot rate')
    plt.title('5-year spot curve')
    for i in range(0,10):
        evaluate_spot(data[i])
        plt.plot(data[i]["t"], label = labels[i])
    plt.legend(bbox_to_anchor = (0.77,0.95), loc='upper left')
    plt.show()
plot_spot(ds)


# Plot the interpolated spot curve
def plot_ip_spot(data):
    plt.xlabel('time to maturity')
    plt.ylabel('spot rate')
    plt.title('5-year interpolated spot curve')
    for i in range(len(data)):
        spot = evaluate_spot(data[i])
        x, y = ip_ytm(data[i]["t"], spot.squeeze())
        plt.plot(x, y, label = labels[i])
    plt.legend(bbox_to_anchor = (0.77,0.95), loc='upper left')
    plt.show()
plot_ip_spot(ds)


# Calculate the forward rate
def evaluate_forward(data):
    y = evaluate_spot(data).squeeze()
    x, y = ip_ytm (data["t"], y)
    f_1 = ((1 + y[3]) ** 2) / (1 + y[1]) - 1
    f_2 = (((1 + y[5]) ** 3) / (1 + y[1])) ** (1/2) - 1
    f_3 = (((1 + y[7]) ** 4) / (1 + y[1])) ** (1/3) - 1
    f_4 = (((1 + y[9]) ** 5) / (1 + y[1])) ** (1/4) - 1
    forwards = [f_1,f_2,f_3,f_4]
    return forwards


# Plot the forward curve
def plot_forward(data):
    plt.xlabel('time to maturity')
    plt.ylabel('forward rate')
    plt.title('1-year forward rate curve')
    for i in range(0,10):
        plt.plot(['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'], evaluate_forward(data[i]), label = labels[i])
    plt.legend(loc = 'upper right', prop={"size":8})
    plt.show()
plot_forward(ds)


# calculate covariance matrix
def find_comatrix(data):
    log = np.zeros([5,9])
    yi = np.zeros([5,10])
    for i in range(len(data)):
        x,y = ip_ytm(data[i]["t"], data[i]["y"])
        yi[0,i] = y[1]
        yi[1,i] = y[3]
        yi[2,i] = y[5]
        yi[3,i] = y[7]
        yi[4,i] = y[9]

    for i in range(0, 9):
        log[0, i] = np.log(yi[0,i+1]/yi[0,i])
        log[1, i] = np.log(yi[1,i+1]/yi[1,i])
        log[2, i] = np.log(yi[2,i+1]/yi[2,i])
        log[3, i] = np.log(yi[3,i+1]/yi[3,i])
        log[4, i] = np.log(yi[4,i+1]/yi[4,i])

    return np.cov(log),log
print(find_comatrix(ds)[0])


def forward_matrix(data):
    m = np.zeros([4,10])
    for index in range(len(data)):
        m[0,index] = evaluate_forward(data[index])
    return m


print(np.cov(forward_matrix(ds)))


evalue1, evec1 = LA.eig(np.cov(forward_matrix(ds)[1]))
print("eigenvalue:", evalue1, "\n eigenvector:", evec1)

evalue2, evec2 = LA.eig(np.cov(forward_matrix(ds)))
print("eigenvalue:", evalue2, "\n eigenvector: ", evec2)