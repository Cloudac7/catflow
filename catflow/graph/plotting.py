import seaborn as sns
from matplotlib import pyplot as plt


def _default_theme():
    theme_dict = {
        "linewidth": 2,
        "fontsize": 24
    }
    return theme_dict


def canvas_style(
        context='notebook',
        style='ticks',
        palette='deep',
        font='sans-serif',
        font_scale=1.5,
        color_codes=True,
        rc=None,
        **kwargs
):
    """set basic properties for canvas

    :param context: select context of the plot. Please refer to seaborn contexts.
    :param style: select style of the plot. Please refer to seaborn styles.
    :param palette: Color palette, see color_palette()
    :param font: Font family, see matplotlib font manager.
    :param font_scale: Separate scaling factor to independently scale the size of the font elements.
    :param color_codes: If True and palette is a seaborn palette,
        remap the shorthand color codes (e.g. “b”, “g”, “r”, etc.) to the colors from this palette.
    :param rc: rc dict to optimize the plot. Please refer to matplotlib document for description in detail.
    """
    sns.set_theme(
        context=context,
        style=style,
        palette=palette,
        font=font,
        font_scale=font_scale,
        color_codes=color_codes,
        rc=rc
    )
    sns.set_style({
        'font.sans-serif': 'DejaVu Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif'
        })


class AxesInit(object):
    def __init__(
            self,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            **kwargs
    ):
        if fig is None:
            self.fig = plt.figure()
        if ax is None:
            self.ax = self.fig.subplot()
        self.kwargs = kwargs

    def add_text(self, title=None, xlabel=None, ylabel=None, fontsize=28):
        self.ax.set_title(title, fontsize=fontsize)
        self.ax.set_xlabel(xlabel, fontsize=fontsize)
        self.ax.set_ylabel(ylabel, fontsize=fontsize)

    def set_theme(self, linewidth=2, label_fontsize=24):
        for item in self.ax.spines:
            self.ax.spines[item].set_linewidth(linewidth)
        for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
            label.set_fontsize(label_fontsize)
        self.ax.tick_params(length=6, width=2)
        self.ax.tick_params(which='minor', length=3, width=2)

def square_grid(num_items):
    import math
    n = math.ceil(math.sqrt(num_items))
    return n
