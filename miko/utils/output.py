import pandas as pd
import seaborn as sns


def canvas_style(
        context='notebook',
        style='darkgrid',
        palette='deep',
        font='sans-serif',
        font_scale=1,
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


def to_csv():
    pass
