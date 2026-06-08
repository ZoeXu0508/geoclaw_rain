"""
World map (ocean layer) with track lines and rectangular areas (of landfall, floodmap etc.)
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import numpy as np

import gcvalid.plot.common as u_plot


class PlotEventsOverview:
    def __init__(self, figure, tracks_w_areas, proj_data=None):
        self.pick_callbacks = []
        self.proj_ax = ccrs.PlateCarree()
        self.proj_data = self.proj_ax if proj_data is None else proj_data
        self.figure = figure
        self.ax = self.figure.add_subplot(111, projection=self.proj_ax)
        self.areas = None
        self.tracks = None
        self.replace_trackset(tracks_w_areas)
        self.ax.spines['geo'].set_linewidth(0.5)
        self.ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)
        self.ax.set_extent((-120, 180, -50, 70), crs=self.proj_data)
        u_plot.ax_add_ticks(self.ax, self.proj_data)
        self.figure.canvas.mpl_connect("pick_event", self.pick_cb)
        self.figure.canvas.mpl_connect("motion_notify_event", self.hover_cb)


    def replace_trackset(self, tracks_w_areas):
        self.tracks_w_areas = tracks_w_areas
        self.track_mask = np.ones((len(self.tracks_w_areas),), dtype=bool)
        self._update_collections()


    def _update_collections(self):
        if self.tracks is not None:
            self.tracks.remove()
        if self.areas is not None:
            self.areas.remove()
        if len(self.tracks_w_areas) == 0:
            return
        self.track_by_area = [
            i_tr for i_tr, tr in enumerate(self.tracks_w_areas)
            for area in tr[1] if self.track_mask[i_tr]]
        self.areas_by_track = [
            [i_area for i_area, area_i_tr in enumerate(self.track_by_area) if area_i_tr == i_tr]
            for i_tr, _ in enumerate(self.tracks_w_areas) if self.track_mask[i_tr]]
        self.tracks = mcollections.LineCollection(
            [tr[0] for i_tr, tr in enumerate(self.tracks_w_areas) if self.track_mask[i_tr]],
            colors='k',
            linewidths=np.full((len(self.tracks_w_areas),), 0.5),
            transform=self.proj_data, picker=True)
        self.tracks.set_pickradius(5)
        self.areas = mcollections.PatchCollection(
            [mpatches.Rectangle(*area)
             for i_tr, tr in enumerate(self.tracks_w_areas)
             for area in tr[1] if self.track_mask[i_tr]],
            facecolor=(0, 0, 0, 0), edgecolor='r',
            linewidths=np.full((len(self.track_by_area),), 0.5),
            transform=self.proj_data, picker=True)
        self.areas.set_pickradius(3)
        self.ax.add_collection(self.tracks)
        self.ax.add_collection(self.areas)


    def hover_cb(self, event):
        if event.inaxes is not self.ax:
            return

        lw_tr = self.tracks.get_linewidths()

        cont_tr, match_tr = self.tracks.contains(event)
        cont_areas, match_areas = self.areas.contains(event)

        # no matches and no hightlights to remove
        if all(w <= 0.5 for w in lw_tr) and not (cont_tr or cont_areas):
            return

        # merge matches into one list
        matches = list(match_tr['ind'])
        matches += [self.track_by_area[i_area] for i_area in match_areas['ind']]
        matches = np.unique(matches)

        # remove matches for invisible items and select the first match
        matches = [m for m in matches if lw_tr[m] > 0]
        i_tr = matches[0] if len(matches) > 0 else -1

        lw_areas = self.areas.get_linewidths()
        lw_areas = [np.clip(w, 0, 0.5) for w in lw_areas]
        lw_tr = [np.clip(w, 0, 0.5) for w in lw_tr]
        if i_tr >= 0:
            lw_tr[i_tr] = 2
            for m in self.areas_by_track[i_tr]:
                lw_areas[m] = 2
        self.areas.set_linewidths(lw_areas)
        self.tracks.set_linewidths(lw_tr)
        self.figure.canvas.draw_idle()


    def pick_cb(self, event):
        if event.artist is self.tracks:
            matches = event.ind
        elif event.artist is self.areas:
            matches = [self.track_by_area[i_area] for i_area in event.ind]
        else:
            return

        lw_tr = self.tracks.get_linewidths()
        matches = [m for m in matches if lw_tr[m] > 0]
        i_tr = matches[0] if len(matches) > 0 else -1
        if i_tr > 0:
            for fun in self.pick_callbacks:
                fun(i_tr)


    def filter(self, mask):
        self.track_mask[:] = mask
        self._update_collections()
        self.figure.canvas.draw_idle()


class PlotSingleEvent:
    def __init__(self, figure):
        self.pick_callbacks = []
        self.figure = figure
        self.ax = None
        self.proj_ax = ccrs.PlateCarree()
        self.areas = None
        self.figure.canvas.mpl_connect("pick_event", self.pick_cb)


    def pick_cb(self, event):
        if event.artist is not self.areas:
            return
        for fun in self.pick_callbacks:
            fun(self.area_ids[event.ind[0]])


    def plot(self, track, areas, proj_data=None):
        proj_data = self.proj_ax if proj_data is None else proj_data
        self.figure.clf()
        self.ax = self.figure.add_subplot(111, projection=self.proj_ax)
        self.ax.spines['geo'].set_linewidth(0.5)
        self.ax.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)

        self.ax.plot(track[:, 0], track[:, 1], color='k', transform=proj_data)
        self.area_ids = [i for i, rect in areas]
        self.areas = mcollections.PatchCollection(
            [mpatches.Rectangle(*rect) for i, rect in areas],
            facecolor=(1, 0, 0, 0.3), edgecolor='r',
            linewidths=np.full((len(areas),), 0.5),
            transform=proj_data, picker=True)
        self.ax.add_collection(self.areas)
        areas_extent_lon = (
            min([xmin for i, ((xmin, ymin), width, height) in areas]),
            max([xmin + width for i, ((xmin, ymin), width, height) in areas]))
        areas_extent_lat = (
            min([ymin for i, ((xmin, ymin), width, height) in areas]),
            max([ymin + height for i, ((xmin, ymin), width, height) in areas]))
        extent_lon = (min(areas_extent_lon[0], track[:, 0].min()) - 5,
                      max(areas_extent_lon[1], track[:, 0].max()) + 5)
        extent_lat = (min(areas_extent_lat[0], track[:, 1].min()) - 5,
                      max(areas_extent_lat[1], track[:, 1].max()) + 5)
        width, height = (extent_lon[1] - extent_lon[0],
                         extent_lat[1] - extent_lat[0])
        aspect_ratio = 1
        width, height = (width if width > height else height / aspect_ratio,
                         height if height > width else width * aspect_ratio)
        mid_lon = 0.5 * sum(extent_lon)
        mid_lat = 0.5 * sum(extent_lat)
        extent = (mid_lon - 0.5 * width, mid_lon + 0.5 * width,
                  mid_lat - 0.5 * height, mid_lat + 0.5 * height)
        self.ax.set_extent(extent, crs=proj_data)
        u_plot.ax_add_ticks(self.ax, proj_data)


    def add_areas(self, areas, proj_data=None):
        proj_data = self.proj_ax if proj_data is None else proj_data
        # additional areas that won't be interactive
        self.areas_added = mcollections.PatchCollection(
            [mpatches.Rectangle(*rect) for rect in areas],
            facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor='k',
            linewidths=np.full((len(areas),), 0.5),
            transform=proj_data)
        self.ax.add_collection(self.areas_added)


    def highlight(self, mask):
        if self.areas is None:
            return
        self.areas.set_linewidths([2 if m else 0.5 for m in mask])
