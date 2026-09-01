"""Publication-scale typography for the Paper 1 plotting notebooks.

The manuscript displays many source PDFs at substantially less than their
native width.  These settings target an effective printed size of about
8--9 pt for titles, 7.5--8 pt for axis labels, and at least 7 pt for ticks,
legends, and colour bars.  Figure-specific layout changes are presentation
only unless explicitly documented in the corresponding notebook.
"""

from __future__ import annotations

import re

import matplotlib.ticker as ticker


CANVAS = {
    "figure2c_ubar_MERRA2NEWNAM": (6.20, 4.60),
    "figure3_h2o": (5.36, 2.55),
    "figure3_clo": (5.36, 2.55),
    "figure04_merra2_2020_waccm0008_nam_o3_context_MERRA2NEWNAM": (5.78, 4.50),
    "figure15f_ubar_N2correction_bootstrap": (5.92, 5.65),
    "figure05a_hindcast_o3_evolution": (5.92, 3.50),
    "figure05b_hindcast_u60n10_tmin50_evolution": (5.92, 3.55),
    "figure06b": (5.92, 5.85),
    "figure07a_u60n10_rmse_vs_o3_rmse": (2.15, 2.05),
    "figure07b_tmin50_rmse_vs_u60n10_rmse": (2.15, 2.05),
    "figure07c_tmin50_rmse_vs_o3_rmse": (2.15, 2.05),
    "figure08b_jan_wave_vs_o3minimum_raw": (3.30, 2.90),
    "figure08h": (3.30, 2.90),
    "figure09d": (5.92, 4.40),
    "figA1": (7.05, 4.35),
    "figA2": (5.92, 6.75),
    "figure01bc_waccm0008_merra2_2020_o3_anomaly_1to100hpa": (5.92, 3.25),
    "figure16b_daily": (5.92, 3.80),
    "figure17a": (5.36, 6.50),
    "figure18a": (5.92, 3.72),
    "figure08a": (5.92, 3.60),
    "figure09a_feb_o3_minimum_date_histogram": (5.92, 3.60),
    "figure11b": (6.35, 3.30),
}


def _style_legend(legend, *, fontsize: float = 7.2, ncols: int | None = None) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(fontsize)
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(max(fontsize, 7.4))
    if ncols is not None:
        legend.set_ncols(ncols)


def _generic_style(figure) -> None:
    if getattr(figure, "_paper1_generic_style_done", False):
        return
    figure._paper1_generic_style_done = True
    if getattr(figure, "_suptitle", None) is not None:
        figure._suptitle.set_fontsize(9.0)
        figure._suptitle.set_fontweight("bold")
    for figure_text in figure.texts:
        if figure_text is not getattr(figure, "_suptitle", None):
            figure_text.set_fontsize(7.4)
    for axis in figure.axes:
        axis.title.set_fontsize(8.4)
        axis.title.set_fontweight("bold")
        axis.xaxis.label.set_fontsize(7.8)
        axis.yaxis.label.set_fontsize(7.8)
        axis.tick_params(
            axis="both", which="major", labelsize=7.0,
            length=2.8, width=0.65, pad=2.0,
        )
        axis.tick_params(
            axis="both", which="minor", labelsize=6.8,
            length=1.8, width=0.55,
        )
        axis.xaxis.get_offset_text().set_fontsize(7.0)
        axis.yaxis.get_offset_text().set_fontsize(7.0)
        for annotation in axis.texts:
            content = annotation.get_text().strip()
            if re.fullmatch(r"\(?[a-zA-Z]\)?", content):
                annotation.set_fontsize(8.6)
                annotation.set_fontweight("bold")
            elif len(content) > 90:
                annotation.set_fontsize(6.8)
            else:
                annotation.set_fontsize(7.0)
        _style_legend(axis.get_legend())
        for line in axis.lines:
            line.set_linewidth(max(float(line.get_linewidth()), 0.8))
            if line.get_marker() not in (None, "None", "", " "):
                line.set_markersize(max(float(line.get_markersize()), 3.7))
    for legend in figure.legends:
        _style_legend(legend)


def apply_paper_style(figure, stem: str) -> None:
    """Apply the accepted large-font layout immediately before saving."""

    if getattr(figure, "_paper1_style_stem", None) == stem:
        return
    figure._paper1_style_stem = stem
    if stem not in CANVAS:
        raise KeyError(f"No publication style registered for {stem!r}")
    figure.set_size_inches(*CANVAS[stem], forward=True)
    _generic_style(figure)

    if stem == "figure2c_ubar_MERRA2NEWNAM":
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text("Observed and modeled precursor relationships")
            figure._suptitle.set_fontsize(8.8)
            figure._suptitle.set_y(0.98)
        for index, axis in enumerate(figure.axes[:6]):
            axis.set_xlabel(
                r"DJF EP100 ($\sigma$)" if index % 3 in (0, 1)
                else "JFMA 50-hPa NAM",
                fontsize=7.2,
            )
        figure.subplots_adjust(
            left=0.10, right=0.985, top=0.90, bottom=0.28,
            wspace=0.22, hspace=0.22,
        )
        handles = []
        for legend in list(figure.legends):
            handles.extend(
                getattr(legend, "legend_handles", getattr(legend, "legendHandles", []))
            )
            legend.remove()
        labels = (
            "Low-O3 quartile", "All-year fit (EP)", "Low-O3 fit (EP)",
            "Fit excluding low-O3 (EP)", "All-year fit (AO-NAM)",
            "Low-O3 fit (AO-NAM)", "MERRA-2 2020", "WACCM year 0008",
            "* p < 0.05",
        )
        if handles:
            figure.legend(
                handles[: len(labels)], labels[: len(handles)],
                loc="lower center", bbox_to_anchor=(0.50, 0.008), ncol=3,
                fontsize=7.6, frameon=True, columnspacing=0.75,
                handletextpad=0.45, handlelength=2.2, borderaxespad=0.0,
            )

    elif stem in {"figure3_h2o", "figure3_clo"}:
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text("")
        is_h2o = stem == "figure3_h2o"
        titles = (
            (r"(a) Aura MLS H$_2$O, 2019/20", r"(b) WACCM H$_2$O, year 0008")
            if is_h2o else
            (r"(a) Aura MLS ClO, 2019/20", r"(b) WACCM ClO$_x$, year 0008")
        )
        for axis, title in zip(figure.axes[:2], titles):
            axis.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
            for index, tick_label in enumerate(axis.get_xticklabels()):
                tick_label.set_visible(index % 2 == 0)
            legend = axis.get_legend()
            if legend is not None:
                for text, label in zip(
                    legend.get_texts(), ("Tmin < 195 K (PSC I)", "Tmin = 195 K")
                ):
                    text.set_text(label)
                    text.set_fontsize(6.8)

    elif stem == "figure04_merra2_2020_waccm0008_nam_o3_context_MERRA2NEWNAM":
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text(r"MERRA-2 2020 and WACCM year 0008 NAM/AO/O$_3$ context")
            figure._suptitle.set_fontsize(8.8)
        data_axes = figure.axes[:6]
        for column, title in enumerate(("MERRA-2 2020", "WACCM year 0008")):
            data_axes[column].set_title(title, fontsize=8.1, fontweight="bold", loc="left")
        for index, axis in enumerate(data_axes):
            row, column = divmod(index, 2)
            if column == 1:
                axis.set_ylabel("")
            elif row == 0:
                axis.set_ylabel("Pressure (hPa)", fontsize=7.4)
            elif row == 1:
                axis.set_ylabel("AO / NAM", fontsize=7.4)
            else:
                axis.set_ylabel(r"O$_3$ anomaly (DU)", fontsize=7.4)

    elif stem == "figure15f_ubar_N2correction_bootstrap":
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text("")
        wanted = [1, 2, 5, 10, 30, 100, 300]
        for axis in figure.axes:
            if "pressure" in axis.yaxis.label.get_text().lower():
                axis.set_yticks(wanted)
                axis.set_yticklabels([str(value) for value in wanted])
                axis.tick_params(axis="y", which="minor", left=False, labelleft=False)
        calendar_axes = [
            axis for axis in figure.axes
            if "event-relative calendar month" in axis.xaxis.label.get_text().lower()
        ]
        if len(calendar_axes) > 1:
            calendar_axes[0].set_xlabel("")
            calendar_axes[0].tick_params(axis="x", labelbottom=False)

    elif stem == "figure05b_hindcast_u60n10_tmin50_evolution":
        for axis, location in zip(figure.axes[:2], ("lower left", "upper left")):
            handles, labels = axis.get_legend_handles_labels()
            if axis.get_legend() is not None:
                axis.get_legend().remove()
            if handles:
                axis.legend(
                    handles, labels, loc=location, fontsize=7.6,
                    frameon=True, framealpha=0.88, borderpad=0.35,
                    labelspacing=0.30, handlelength=2.2,
                )

    elif stem in {
        "figure05a_hindcast_o3_evolution", "figure06b", "figure16b_daily"
    }:
        for legend in list(figure.legends):
            _style_legend(legend, fontsize=7.8)
        for axis in figure.axes:
            _style_legend(axis.get_legend(), fontsize=7.8)
        if stem == "figure16b_daily" and figure.axes:
            for annotation in figure.axes[0].texts:
                if annotation.get_text().strip() == "(a)":
                    annotation.set_position((0.01, 0.78))

    elif stem.startswith("figure07"):
        for axis in figure.axes:
            axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))
            axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))

    elif stem == "figure08b_jan_wave_vs_o3minimum_raw":
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text("")
        for axis in figure.axes:
            for annotation in axis.texts:
                if re.fullmatch(r"(?:ref )?\d{3,4}", annotation.get_text().strip()):
                    annotation.set_fontsize(6.2)

    elif stem == "figA2":
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text("Precursor sensitivity to calendar windows")
            figure._suptitle.set_fontsize(8.8)
            figure._suptitle.set_y(0.985)
        axes = figure.axes[:6]
        if len(axes) != 6:
            raise RuntimeError(f"Appendix A2 expected six data axes, found {len(axes)}")
        placements = {
            0: (0.105, 0.700, 0.350, 0.205), 3: (0.535, 0.700, 0.350, 0.205),
            1: (0.105, 0.405, 0.350, 0.205), 4: (0.535, 0.405, 0.350, 0.205),
            2: (0.105, 0.110, 0.350, 0.205), 5: (0.535, 0.110, 0.350, 0.205),
        }
        titles = {
            0: "(a) MERRA-2\nEP100 vs 50-hPa NAM",
            3: "(b) WACCM\nEP100 vs 50-hPa NAM",
            1: "(c) MERRA-2\nEP100 vs O$_3$ minimum",
            4: "(d) WACCM\nEP100 vs O$_3$ minimum",
            2: "(e) MERRA-2\n50-hPa NAM vs surface AO",
            5: "(f) WACCM\n50-hPa NAM vs surface AO",
        }
        for index, axis in enumerate(axes):
            axis.set_position(placements[index])
            axis.set_title(titles[index], fontsize=7.8, fontweight="bold", pad=2)
            for annotation in list(axis.texts):
                content = annotation.get_text().strip()
                if content in {"MERRA2", "WACCM (INT-3D-ETH)"}:
                    annotation.remove()
                elif index in (2, 5) and content in {"JFMA", "FMA", "MA", "A"}:
                    annotation.remove()
                elif ("->" in content or "→" in content) and len(content) >= 7:
                    annotation.set_fontsize(5.7)
        axes[0].set_ylabel("Lagged windows", fontsize=7.3)
        axes[3].set_ylabel("")
        axes[1].set_ylabel("EP100 window", fontsize=7.3)
        axes[4].set_ylabel("")
        for index in (2, 5):
            axes[index].set_ylabel("")
            axes[index].tick_params(axis="y", labelleft=False)
            axes[index].set_xticks(
                range(4), ("JFMA-JFMA", "FMA-FMA", "MA-MA", "A-A")
            )
            axes[index].tick_params(axis="x", labelsize=6.4, pad=2)
            axes[index].set_xlabel("NAM window - AO window", fontsize=7.0)
        figure.text(
            0.50, 0.018,
            r"Cell: standardized slope $\beta$ (all years; excluding low-O$_3$ years in parentheses); * $p<0.05$.",
            ha="center", va="bottom", fontsize=6.8,
        )
        if len(figure.axes) > 6:
            colourbar_axis = figure.axes[6]
            colourbar_axis.set_position((0.915, 0.145, 0.018, 0.700))
            colourbar_axis.tick_params(labelsize=6.8)
            colourbar_axis.yaxis.label.set_fontsize(7.2)

    elif stem == "figure01bc_waccm0008_merra2_2020_o3_anomaly_1to100hpa":
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text("")
        for index, (axis, title) in enumerate(
            zip(figure.axes[:2], ("(a) WACCM year 0008", "(b) MERRA-2 2020"))
        ):
            axis.set_title(title, fontsize=8.0, fontweight="bold", pad=3)
            axis.set_xlabel("Calendar month", fontsize=7.5)
            axis.set_ylabel("Pressure (hPa)" if index == 0 else "", fontsize=7.5)
            axis.tick_params(axis="both", labelsize=7.0)
        if len(figure.axes) > 2:
            colourbar_axis = figure.axes[2]
            colourbar_axis.tick_params(labelsize=6.8)
            colourbar_axis.yaxis.label.set_fontsize(7.2)

    elif stem == "figure17a":
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text("Reference and hindcast vertical NAM evolution")
            figure._suptitle.set_fontsize(8.8)
        titles = (
            "(a) Year 0008 reference", "(b) January-initialized mean",
            "(c) February-initialized mean", "(d) March-initialized mean",
        )
        for axis, title in zip(figure.axes[:4], titles):
            axis.set_title(title, fontsize=7.7, fontweight="bold", loc="left", pad=2)

    elif stem == "figure18a":
        for axis in figure.axes:
            legend = axis.get_legend()
            if legend is not None:
                for text, label in zip(
                    legend.get_texts(),
                    ("January", "February", "March", "Year 0008", "Climatology"),
                ):
                    text.set_text(label)
                legend.set_ncols(3)

    elif stem == "figure08a":
        for axis in figure.axes:
            axis.set_title(
                r"January initialization: O$_3$-minimum dates (5-day boxes)",
                fontsize=8.2, fontweight="bold",
            )
            axis.set_xlabel(r"Centered 5-day O$_3$-minimum date", fontsize=7.5)

    elif stem == "figure09a_feb_o3_minimum_date_histogram":
        for axis in figure.axes:
            axis.set_title(
                r"February initialization: O$_3$-minimum dates (5-day boxes)",
                fontsize=8.2, fontweight="bold",
            )
            axis.set_xlabel(r"Centered 5-day O$_3$-minimum date", fontsize=7.5)

    elif stem == "figure11b":
        if getattr(figure, "_suptitle", None) is not None:
            figure._suptitle.set_text("")
        for legend in list(figure.legends):
            legend.remove()
        for index, axis in enumerate(figure.axes[:2]):
            axis.set_xlabel("Window start (days after initialization)", fontsize=7.2)
            axis.set_ylabel("Averaging length (days)" if index == 0 else "", fontsize=7.2)
        figure.subplots_adjust(
            left=0.08, right=0.91, top=0.86, bottom=0.16, wspace=0.08
        )

    figure.canvas.draw_idle()
