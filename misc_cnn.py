import os
import socket
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
from keras.callbacks import Callback


# #############################
# Canary Interrupt
# #############################
class CanaryInterruptCallback(Callback):

    def __init__(self, path: str, starts_active: bool = True, label: str = None, out_stream=sys.stdout):
        super().__init__()

        self.active: bool = starts_active
        self.label: str = label
        self.shutdown_source: bool = False
        self.out_stream = out_stream

        os.makedirs(path, exist_ok=True)

        self.__canary_file = path + os.sep + 'canary_interrupt.txt'
        if os.path.exists(self.__canary_file):
            os.remove(self.__canary_file)

        f = open(self.__canary_file, 'w')
        f.write(
            'Canary interrupt for CNN training started at ' + gct() + '.\nDelete this file to safely stop your '
                                                                      'training.')
        if label is not None:
            f.write('\nLabel: ' + str(label).strip())
        f.write('\n\nCreated by Nils Foerster.')
        f.close()

        print('Placed canary file here:' + str(self.__canary_file), file=self.out_stream)

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(logs)
        if self.active:
            if not os.path.exists(self.__canary_file):
                print('Canary file not found! Shutting down training!', file=self.out_stream)
                self.shutdown_source = True
                self.model.stop_training = True

    def on_train_end(self, logs={}):
        super().on_train_end(logs)
        if os.path.exists(self.__canary_file):
            os.remove(self.__canary_file)


# #############################
# Live Plotting
# #############################
class PlotTrainingLiveCallback(Callback):
    # packages required: os, socket, matplotlib as plt

    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff']

    def __init__(self, out_dir: str, out_extensions: [str] = ['.png', '.pdf', '.svg', '.csv', '.tex'],
                 label: str = None, save_timestamps: bool = True, epochs_target: int = None,
                 metrics_names: [str] = None, plot_eta_extra: bool = True, plot_dpi: int = 400,
                 plot_transparancy: bool = True):
        super().__init__()
        self.label = label
        self.out_extensions = out_extensions
        self.metrics_names = metrics_names
        self.plot_dpi = plot_dpi
        self.plot_transparancy = plot_transparancy
        self.save_timestamps = save_timestamps
        self.epochs_target = epochs_target
        self.plot_eta_extra = plot_eta_extra
        self.out_dir = out_dir

        self.live_plot_dir = out_dir + 'live_plot' + os.sep
        self.timestamp_file_name = self.live_plot_dir + 'training_timestamps.csv'
        os.makedirs(self.live_plot_dir, exist_ok=True)

        if os.path.exists(self.timestamp_file_name):
            os.remove(self.timestamp_file_name)

        self.epoch_start_timestamp = time.time()
        self.epoch_duration_list = []
        self.host_name = str(socket.gethostname())

        self.epochCount = 0
        self.history = {}

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs)
        self.write_timestamp_line('Training start;' + gct())
        self.write_timestamp_line('Epoch;Timestamp')

        if self.metrics_names is None:
            self.metrics_names = self.model.metrics_names

        for metric in self.metrics_names:
            self.history[metric] = []
            self.history['val_' + metric] = []

    def on_train_end(self, logs={}):
        super().on_train_end(logs)
        self.write_timestamp_line('Training finished;;' + gct())
        self.plot_training_time()
        self.plot_training_history_live()

    def on_epoch_begin(self, epoch, logs={}):
        super().on_epoch_begin(logs)
        self.epochCount = self.epochCount + 1
        self.epoch_start_timestamp = time.time()
        self.write_timestamp_line()

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(logs)
        t = int(time.time() - self.epoch_start_timestamp)
        self.epoch_duration_list.append(t)

        for metric in self.metrics_names:
            val = 'val_' + metric
            self.history[metric].append(logs[metric])
            self.history[val].append(logs[val])

        self.plot_training_time()
        self.plot_training_history_live()

        if self.plot_eta_extra:
            self.plot_training_time(p_out_dir=self.out_dir, png_only=True)

    def write_timestamp_line(self, line=None):
        if not self.save_timestamps:
            return

        try:
            f = open(self.timestamp_file_name, 'a')
            if line is None:
                line = str(self.epochCount) + ';' + gct()

            f.write(line + '\n')
            f.close()
        except Exception as e:
            # TODO print stacktrace
            pass

    def plot_training_time(self, p_out_dir: str = None, png_only: bool = False):
        # Plotting epoch duration
        if p_out_dir is None:
            p_out_dir = self.live_plot_dir

        for extension in self.out_extensions:
            if png_only:
                extension = '.png'

            self.save_metric(data_name='training_time', extension=extension, title='Model Training Time',
                             data1=self.epoch_duration_list, y_label='Duration (Sec.)', p_out_dir=p_out_dir)

    def save_metric(self, data_name: str, title: str, extension: str, data1: [float], data2: [float] = None,
                    y_label: str = None, p_out_dir: str = None):
        if p_out_dir is None:
            p_out_dir = self.live_plot_dir

        extension = extension.lower().strip()
        if not extension.startswith('.'):
            extension = '.' + extension

        if extension == '.csv':
            self.save_csv_metric(data_name=data_name, data_label=y_label, data1=data1, data2=data2)
            return
        if extension == '.tex':
            self.save_tex_metric(data_name=data_name, title=title, data_label=y_label, data1=data1, data2=data2)
            return

        if self.label is not None:
            title = title + ' [' + self.label + ']'

        plt.title(title)
        plt.ylabel(data_name)
        if y_label is not None:
            plt.ylabel(y_label)
        plt.xlabel('Epoch')
        plt.plot(data1)

        if data2 is not None:
            plt.plot(data2)
            plt.legend(['Train', 'Validation'], loc='best')

        plt.savefig(p_out_dir + data_name + '_live' + extension, dpi=self.plot_dpi,
                    transparent=self.plot_transparancy)
        plt.clf()

    def save_csv_metric(self, data_name: str, data_label: str, data1, data2=None):
        f_name = self.live_plot_dir + data_name + '_live.csv'
        f = open(f_name, 'w')

        f.write('Epoch;' + data_label)
        if data2 is not None:
            f.write(';Validation ' + data_label)
        f.write(';\n')

        for i in range(len(data1)):
            f.write(str(i + 1) + ';' + str(data1[i]))
            if data2 is not None:
                f.write(';' + str(data2[i]))
            f.write(';\n')

    def save_tex_metric(self, data_name: str, title: str, data_label: str, data1, data2=None):
        data = [data1]
        titles = ['Training']
        colors = ['blue']

        min_y = min(data1)
        max_y = max(data1)

        if data2 is not None:
            data.append(data2)
            titles.append('Validation')
            colors.append('orange')

            min_y = min(min_y, min(data2))
            max_y = max(max_y, max(data2))

        min_y = max(min_y - 0.1337, 0)
        max_y = min(max_y + 0.1337, 1)

        out_text = get_plt_as_tex(data_list_y=data, plot_titles=titles, plot_colors=colors, title=title,
                                  label_y=data_label, max_x=len(data1), min_x=1, max_y=max_y, min_y=min_y)

        f_name = self.live_plot_dir + data_name + '_live.tex'
        f = open(f_name, 'w')
        f.write(out_text)
        f.close()

        return out_text

    def plot_training_history_live(self):
        # Plotting epoch duration
        for metric in self.metrics_names:
            val = 'val_' + metric
            m = metric.capitalize()
            title = 'Model: ' + m

            for extension in self.out_extensions:
                data1 = self.history[metric]
                data2 = None
                if val in self.history:
                    data2 = self.history[val]

                self.save_metric(data_name=metric, extension=extension, title=title, y_label=m, data1=data1,
                                 data2=data2)


# ###############################
# OTHER UTIL FUNCTIONS
# ###############################

def gct(raw: bool = False):
    n = datetime.now()
    if raw:
        return n
    return n.strftime("%d/%m/%Y %H:%M:%S")


def get_time_diff(start_time: datetime):
    diff = datetime.now() - start_time
    minutes = divmod(diff.total_seconds(), 60)

    m: str = str(int(minutes[0]))
    s: str = str(int(minutes[1]))
    if minutes[1] < 10:
        s = '0' + s
    return m + ':' + s


# ###############################
# API TO LaTeX TIKZ
# ###############################

def create_tikz_axis(title: str, label_y: str, label_x: str = 'Epoch', max_x: float = 1.0, min_x: float = 0.0,
                     max_y: float = 1.0, min_y: float = 0.0, tick_count: int = 10, legend_pos: str = 'north west'):
    max_x = float(max_x)
    max_y = float(max_y)
    tick_count = float(tick_count)

    tick_x = max_x / tick_count
    tick_y = max_y / tick_count
    if min_x + max_x > 10:
        tick_x = int(tick_x)
    if min_y + max_y > 10:
        tick_y = int(tick_y)

    axis_text = '\\begin{center}\n\t\\begin{tikzpicture}\n\t\\begin{axis}[title={' + title + '},xlabel={' + label_x + '},ylabel={' + label_y + '},xtick distance=' + str(
        tick_x) + ',ytick distance=' + str(tick_y) + ',xmin=' + str(min_x) + ',xmax=' + str(
        max_x) + ',ymin=' + str(min_y) + ',ymax=' + str(
        max_y) + ',major grid style={line width=.2pt,draw=gray!50},grid=both,height=8cm,width=8cm'
    if legend_pos is not None:
        axis_text = axis_text + ', legend pos=' + legend_pos
    axis_text = axis_text + ']'
    return axis_text


def get_plt_as_tex(data_list_y: [[float]], plot_colors: [], title: str, label_y: str, data_list_x: [[float]] = None,
                   plot_titles: [str] = None,
                   label_x: str = 'Epoch', max_x: float = 1.0, min_x: float = 0.0, max_y: float = 1.0,
                   min_y: float = 0.0, max_entries: int = 4000, legend_pos: str = 'north west'):
    out_text = create_tikz_axis(title=title, label_y=label_y, label_x=label_x, max_x=max_x, min_x=min_x, max_y=max_y,
                                min_y=min_y, legend_pos=legend_pos) + '\n'
    line_count = len(data_list_y[0])
    data_x = None
    steps = int(max(len(data_list_y) / max_entries, 1))

    for j in range(0, len(data_list_y), steps):
        data_y = data_list_y[j]
        if data_list_x is not None:
            data_x = data_list_x[j]

        color = plot_colors[j]

        out_text = out_text + '\t\t\\addplot[color=' + color + '] coordinates {' + '\n'
        for i in range(line_count):
            y = data_y[i]

            x = i + 1
            if data_x is not None:
                x = data_x[i]

            out_text = out_text + '\t\t\t(' + str(x) + ',' + str(y) + ')\n'
        out_text = out_text + '\t\t};\n'

        if plot_titles is not None:
            plot_title = plot_titles[j]
            out_text = out_text + '\t\t\\addlegendentry{' + plot_title + '}\n'

    out_text = out_text + '\t\\end{axis}\n\t\\end{tikzpicture}\n\\end{center}'
    return out_text


if __name__ == "__main__":
    print('There are some util functions for everyone to use within this file. Enjoy. :)')
