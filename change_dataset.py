import csv
import numpy as np

x = np.genfromtxt('./GestureDataset_Padded.csv', delimiter=',')
xis = x[:, 2:]
user_ids = x[:, 1]
labels = x[:, 0] - 1
x_acc_noise = xis[:, 0:100]
y_acc_noise = xis[:, 100:200]
z_acc_noise = xis[:, 200:300]
x_gyr_noise = xis[:, 300:400]
y_gyr_noise = xis[:, 400:500]
z_gyr_noise = xis[:, 500:600]
print z_gyr_noise.shape
print x_acc_noise.shape
print y_acc_noise.shape
print xis.shape

with open('x_acc_noise.csv', 'w') as gdata:
    np.savetxt(gdata, x_acc_noise, delimiter=',')
gdata.close()

with open('y_acc_noise.csv', 'w') as gdata:
    np.savetxt(gdata, y_acc_noise, delimiter=',')
gdata.close()

with open('z_acc_noise.csv', 'w') as gdata:
    np.savetxt(gdata, z_acc_noise, delimiter=',')
gdata.close()

with open('x_gyr_noise.csv', 'w') as gdata:
    np.savetxt(gdata, x_gyr_noise, delimiter=',')
gdata.close()

with open('y_gyr_noise.csv', 'w') as gdata:
    np.savetxt(gdata, y_gyr_noise, delimiter=',')
gdata.close()

with open('z_gyr_noise.csv', 'w') as gdata:
    np.savetxt(gdata, z_gyr_noise, delimiter=',')
gdata.close()

with open('labels.csv', 'w') as gdata:
    np.savetxt(gdata, labels, delimiter=',')
gdata.close()

with open('user_ids.csv', 'w') as gdata:
    np.savetxt(gdata, user_ids, delimiter=',')
gdata.close()


with open('./GestureDataset.csv', 'r') as gdata:
    for row in gdata:
        row = row.split(',')[2:]
        x_acc_raw = np.atleast_2d(np.array(map(float, row[0:len(row)/6])))
        y_acc_raw = np.atleast_2d(np.array(map(float, row[1 * len(row)/6:2 * len(row) / 6])))
        z_acc_raw = np.atleast_2d(np.array(map(float, row[2 * len(row)/6:3 * len(row) / 6])))
        x_gyr_raw = np.atleast_2d(np.array(map(float, row[3 * len(row)/6:4 * len(row) / 6])))
        y_gyr_raw = np.atleast_2d(np.array(map(float, row[4 * len(row)/6:5 * len(row) / 6])))
        z_gyr_raw = np.atleast_2d(np.array(map(float, row[5 * len(row)/6:6 * len(row) / 6])))

        with open('x_acc_raw.csv', 'a') as gdata1:
            np.savetxt(gdata1, x_acc_raw, delimiter=',')
        gdata1.close()

        with open('y_acc_raw.csv', 'a') as gdata2:
            np.savetxt(gdata2, y_acc_raw, delimiter=',')
        gdata2.close()

        with open('z_acc_raw.csv', 'a') as gdata3:
            np.savetxt(gdata3, z_acc_raw, delimiter=',')
        gdata3.close()

        with open('x_gyr_raw.csv', 'a') as gdata4:
            np.savetxt(gdata4, x_gyr_raw, delimiter=',')
        gdata4.close()

        with open('y_gyr_raw.csv', 'a') as gdata5:
            np.savetxt(gdata5, y_gyr_raw, delimiter=',')
        gdata5.close()

        with open('z_gyr_raw.csv', 'a') as gdata6:
            np.savetxt(gdata6, z_gyr_raw, delimiter=',')
        gdata6.close()
gdata.close()
# for row in xis:
#     x_acc_raw = np.vstack(x_acc_raw, row[0:len(row)/6])
#     y_acc_raw = np.vstack(y_acc_raw, row[1 * len(row)/6:2 * len(row) / 6])
#     z_acc_raw = np.vstack(z_acc_raw, row[2 * len(row)/6:3 * len(row) / 6])
#     x_gyr_raw = np.vstack(x_gyr_raw, row[3 * len(row)/6:4 * len(row) / 6])
#     y_gyr_raw = np.vstack(y_gyr_raw, row[4 * len(row)/6:5 * len(row) / 6])
#     z_gyr_raw = np.vstack(z_gyr_raw, row[5 * len(row)/6:6 * len(row) / 6])

# with open('x_acc_raw.csv', 'a') as gdata:
#     np.savetxt(gdata, x_acc_raw, delimiter=',')
# gdata.close()

# with open('y_acc_raw.csv', 'a') as gdata:
#     np.savetxt(gdata, y_acc_raw, delimiter=',')
# gdata.close()

# with open('z_acc_raw.csv', 'a') as gdata:
#     np.savetxt(gdata, z_acc_raw, delimiter=',')
# gdata.close()

# with open('x_gyr_raw.csv', 'a') as gdata:
#     np.savetxt(gdata, x_gyr_raw, delimiter=',')
# gdata.close()

# with open('y_gyr_raw.csv', 'a') as gdata:
#     np.savetxt(gdata, y_gyr_raw, delimiter=',')
# gdata.close()

# with open('z_gyr_raw.csv', 'a') as gdata:
#     np.savetxt(gdata, z_gyr_raw, delimiter=',')
# gdata.close()

# with open('labels.csv', 'a') as gdata:
#     np.savetxt(gdata, labels, delimiter=',')
# gdata.close()

# with open('user_ids.csv', 'a') as gdata:
#     np.savetxt(gdata, user_ids, delimiter=',')
# gdata.close()
