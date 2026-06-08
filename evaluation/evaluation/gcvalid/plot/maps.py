
import cartopy.crs as ccrs
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import numpy as np

import gcvalid.plot.common as u_plot


class PlotMap:
    def __init__(self, figure):
        self.pick_callbacks = []
        self.proj_ax = ccrs.PlateCarree()
        self.proj_data = self.proj_ax
        self.gauges = []
        self.figure = figure
        self.figure.canvas.mpl_connect("pick_event", self.pick_cb)


    def pick_cb(self, event):
        selected_set = [i for i, g in enumerate(self.gauges) if event.artist is g]
        if len(selected_set) == 0:
            return
        for fun in self.pick_callbacks:
            fun((selected_set[0], event.ind[0]))


    def plot(self, extent, images, proj_data=None):
        self.gauges = []
        self.proj_data = self.proj_data if proj_data is None else proj_data
        self.figure.clf()
        self.ax = self.figure.add_subplot(111, projection=self.proj_ax)
        self.ax.spines['geo'].set_linewidth(0.5)
        for img in images:
            self.ax.imshow(img, origin='upper', extent=extent)
        self.ax.coastlines(linewidth=0.5)
        self.ax.set_extent(extent, crs=self.proj_data)
        u_plot.ax_add_ticks(self.ax, self.proj_data)
        self.figure.canvas.draw_idle()


    def add_gauges(self, locations, kwargs=None, highlight_mask=None):
        kwargs = {} if kwargs is None else kwargs
        self.gauges.append(self.ax.scatter(
            locations[:, 0], locations[:, 1],
            transform=self.proj_data, picker=True, **kwargs))
        self.gauges[-1].set_pickradius(3)
        if highlight_mask is not None:
            color = None
            for key in ['color', 'facecolor', 'facecolors', 'edgecolor', 'edgecolors']:
                if key in kwargs:
                    color = kwargs[key]
                    break
            self.ax.scatter(locations[highlight_mask, 0], locations[highlight_mask, 1],
                            marker="s", s=190, facecolors='none',
                            edgecolors=color, transform=self.proj_data)
            self.figure.canvas.draw_idle()


    def add_areas(self, areas, proj_data=None):
        proj_data = self.proj_ax if proj_data is None else proj_data
        # additional areas that won't be interactive
        self.areas_added = mcollections.PatchCollection(
            [mpatches.Rectangle(*rect) for rect in areas],
            facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor='k',
            linewidths=np.full((len(areas),), 0.5),
            transform=proj_data)
        self.ax.add_collection(self.areas_added)
