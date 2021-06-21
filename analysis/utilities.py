
import hashlib
import pathlib
import urllib.request
import zipfile
from typing import List

import numpy as np
import tqdm
import yaml
import pandas as pd
from plotly import graph_objects
from plotly.colors import DEFAULT_PLOTLY_COLORS, qualitative, convert_colors_to_same_type
from ipywidgets import widgets
from IPython.display import display
from formable.yielding import YieldFunction
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit


STRAIN_PATH_LEGEND_NAMES = {
    'uniaxial': 'Uniaxial',
    'plane_strain': 'Plane strain',
    'biaxial': 'Biaxial',
    'biaxial_1': 'Biaxial 1',
    'biaxial_2': 'Biaxial 2',
    'biaxial_3': 'Biaxial 3',
    'equi_biaxial': 'Equi-biaxial',
}


class WorkHardeningExplorer:

    def __init__(self, hardening_data, show_interpolation, show_non_extrapolated_stress,
                 show_fitted_data, layout_args, figure, buttons):

        self.hardening_data = hardening_data
        self.show_interpolation = show_interpolation
        self.show_non_extrapolated_stress = show_non_extrapolated_stress
        self.show_fitted_data = show_fitted_data
        self.layout_args = layout_args
        self.figure = figure
        self.buttons = buttons

        self.buttons.observe(self.on_extrapolation_type_clicked, 'value')

    def on_extrapolation_type_clicked(self, change):
        extrap_mode = change['new']
        plt_data = []
        for strain_path_idx, (strain_path, hard_data) in enumerate(self.hardening_data.items()):
            plt_data.extend(
                get_extrapolation_mode_plot_data(
                    strain_path,
                    strain_path_idx,
                    hard_data,
                    extrap_mode,
                    STRAIN_PATH_LEGEND_NAMES,
                    self.show_interpolation,
                    self.show_non_extrapolated_stress,
                    self.show_fitted_data,
                )
            )
        with self.figure.batch_update():
            for idx, dat in enumerate(self.figure.data):
                dat.update(plt_data[idx])

    @property
    def visual(self):
        explorer_widgets = widgets.VBox(
            children=[
                self.buttons,
                self.figure,
            ]
        )
        return explorer_widgets


class FormingLimitExplorer:

    def __init__(self, figure, show_forming_limits_toggle_button,
                 show_strain_paths_toggle_button, FLC_workflows, show_3D=False):
        self.figure = figure
        self.show_forming_limits_toggle_button = show_forming_limits_toggle_button
        self.show_strain_paths_toggle_button = show_strain_paths_toggle_button
        self.FLC_workflows = FLC_workflows
        self.show_3D = show_3D

        self.show_forming_limits_toggle_button.observe(
            self._on_toggle_forming_limits, 'value')
        self.show_strain_paths_toggle_button.observe(
            self._on_toggle_strain_paths, 'value')

    def _on_toggle_forming_limits(self, change):

        visibility = {}
        for trace in self.figure.data:
            if trace.name not in visibility or trace.visible in [True, 'legendonly']:
                visibility.update({trace.name: trace.visible})

        with self.figure.batch_update():

            for trace_idx, trace in enumerate(self.figure.data):

                if trace.meta['type'] in ['forming_limit', 'strain_path']:
                    showlegend = False
                else:
                    showlegend = None
                visible = None

                if trace.meta['type'] == 'forming_limit':
                    visible = (
                        visibility[trace.name] or True) if change['new'] is True else False
                    showlegend = change['new']

                elif trace.meta['type'] == 'strain_path':
                    if trace.meta['strain_path_idx'] == 0 and trace.visible:
                        showlegend = not change['new']

                if showlegend is not None:
                    trace.showlegend = showlegend
                if visible is not None:
                    trace.visible = visible

    def _on_toggle_strain_paths(self, change):

        visibility = {}
        for trace in self.figure.data:
            if trace.name not in visibility or trace.visible in [True, 'legendonly']:
                visibility.update({trace.name: trace.visible})

        with self.figure.batch_update():

            for trace_idx, trace in enumerate(self.figure.data):

                if trace.meta['type'] in ['forming_limit', 'strain_path']:
                    showlegend = False
                else:
                    showlegend = None
                visible = None

                if trace.meta['type'] == 'forming_limit':
                    showlegend = not change['new']

                elif trace.meta['type'] == 'strain_path':
                    visible = (
                        visibility[trace.name] or True) if change['new'] is True else False
                    if trace.meta['strain_path_idx'] == 0 and visible:
                        showlegend = change['new']

                if showlegend is not None:
                    trace.showlegend = showlegend
                if visible is not None:
                    trace.visible = visible

    def show(self):
        return self.visual

    @property
    def visual(self):
        visualiser_widgets = widgets.VBox(
            children=[
                widgets.HBox(
                    children=[
                        self.show_forming_limits_toggle_button,
                        self.show_strain_paths_toggle_button,
                    ]
                ),
                self.figure,
            ]
        )
        return visualiser_widgets


def get_color(name: str, sample_names: List[str], normed: bool = True) -> tuple:
    """Returns a RGB tuple for the dataset based on its label, using a qualitative color scale."""
    index = sample_names.index(name.split("_")[0])
    rgb_color_str = convert_colors_to_same_type(qualitative.D3[index])[0][0]
    rgb_color_tuple = tuple([
        (int(i) / (255 if normed else 1))
        for i in rgb_color_str.split('(')[1].split(')')[0].split(',')
    ])

    return rgb_color_tuple


def get_file_from_url(data_folder: str, url: str, name: str, md5: str = None) -> pathlib.Path:
    """Download a file from a URL if it is not already present in `data_folder`. Return
    the local path to the downloaded file.
    """
    data_folder = pathlib.Path(data_folder)
    if not data_folder.is_dir():
        data_folder.mkdir()
    dest_path = data_folder / name
    out_path = dest_path
    if not dest_path.exists():
        tqdm_description = f"Downloading file \"{name}\""
        with tqdm.tqdm(desc=tqdm_description, unit="bytes", unit_scale=True) as t:
            urllib.request.urlretrieve(url, dest_path, reporthook=tqdm_hook(t))

    if md5:
        if not validate_checksum(dest_path, md5):
            raise AssertionError('MD5 does not match: workflow file is corrupt. '
                                 'Delete workflow file and retry download.')
        else:
            print("MD5 validated. Download complete.")

    return out_path


def unzip_file(file_path: str, destination_path: str):
    """Unzip the file at `file_path` to a folder at `destination_path`."""
    print('Unzipping...', end='')
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)
    print('complete.')


def tqdm_hook(t: tqdm.tqdm):
    """Wraps tqdm progress bar to provide update hook method for `urllib.urlretrieve`."""
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def validate_checksum(file_path: pathlib.Path, valid_md5: str) -> bool:
    with open(file_path, 'rb') as binary_zip:
        md5_hash = hashlib.md5()
        md5_hash.update(binary_zip.read())
        digest = md5_hash.hexdigest()
        if digest == valid_md5:
            return True
        else:
            return False


def read_data_yaml(data_yaml_path: str) -> dict:
    path = pathlib.Path(data_yaml_path)
    with open(path) as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def pretty_print_single_crystal_parameters(parameters):
    'Pretty print the single crystal parameter dict'
    out = (
        f'Hardening coefficient,   h_0_sl_sl: {parameters["Al"]["h_0_sl_sl"]/1e6:>7.1f} MPa\n'
        f'Initial CRSS,              xi_0_sl: {parameters["Al"]["xi_0_sl"][0]/1e6:>7.1f} MPa\n'
        f'Final CRSS,              xi_inf_sl: {parameters["Al"]["xi_inf_sl"][0]/1e6:>7.1f} MPa\n'
        f'Hardening exponent,           a_sl: {parameters["Al"]["a_sl"]:>7.1f}\n'
    )
    print(out)


def plot_static_figure_single_crystal_fitting(lm_fitter):
    data = [
        {
            'x': lm_fitter.optimisations[0].get_exp_strain(),
            'y': lm_fitter.optimisations[0].get_exp_stress() / 1e6,
            'mode': 'lines',
            'name': 'Experimental',
            'line': {
                'dash': 'dash',
                'width': 3,
                'color': DEFAULT_PLOTLY_COLORS[0],
            },
        },
        {
            'x': lm_fitter.optimisations[0].get_sim_strain(0),
            'y': lm_fitter.optimisations[0].get_sim_stress(0) / 1e6,
            'mode': 'lines',
            'line': {
                'dash': 'dot',
                'width': 1,
                'color': DEFAULT_PLOTLY_COLORS[1],
            },
            'name': 'Sim. initial iter.',
        },
        {
            'x': lm_fitter.optimisations[-1].get_sim_strain(0),
            'y': lm_fitter.optimisations[-1].get_sim_stress(0) / 1e6,
            'mode': 'lines',
            'name': 'Sim. final iter.',
            'line': {
                'width': 1,
                'color': DEFAULT_PLOTLY_COLORS[1],
            },
        }
    ]
    layout = {
        'width': 280,
        'height': 250,
        'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
        'template': 'simple_white',
        'xaxis': {
            'title': r'True strain, \strain{}',
            'range': [
                -0.01,
                lm_fitter.optimisations[0].get_exp_strain()[-1],
            ],
            'mirror': 'ticks',
            'ticks': 'inside',
            'dtick': 0.05,
            'tickformat': '.2f',
        },
        'yaxis': {
            'title': r'True stress, \stress{} (\MPa)',
            'mirror': 'ticks',
            'ticks': 'inside',
            'range': [0, 310],
            'dtick': 50,
        },
        'legend': {
            'x': 0.95,
            'y': 0.05,
            'xanchor': 'right',
            'yanchor': 'bottom',
            'tracegroupgap': 0,
        },
    }
    fig = graph_objects.Figure(data=data, layout=layout)
    return fig


def plot_static_figure_yield_function_type_comparison(all_load_responses, yield_point):
    yld_funcs_subset = []
    for resp in all_load_responses:
        yield_function_idx = np.where(
            resp.yield_point_criteria[0].values[0] == yield_point)[0][0]
        yld_funcs_subset.append(
            resp.fitted_yield_functions[yield_function_idx]['yield_function']
        )
    plane = [0, 0, 1]
    fig_wig = YieldFunction.compare_2D(yld_funcs_subset, plane)
    fig_wig.layout.annotations = ()  # remove annotations

    line_styles = {
        'Hill1948': {'dash': 'dash'},
        'Barlat_Yld91': {'dash': 'dot'},
        'Barlat_Yld2004_18p': {'dash': 'solid'},
    }
    for idx, trace in enumerate(fig_wig.data):
        if 'VonMises' in trace.name:
            new_name = 'Von Mises'
        elif 'Barlat_Yld91' in trace.name:
            new_name = 'Barlat Yld91'
            trace.line.update(line_styles['Barlat_Yld91'])
        elif 'Barlat_Yld2000_2D' in trace.name:
            new_name = 'Barlat Yld2000-2D'
        elif 'Barlat_Yld2004_18p' in trace.name:
            new_name = 'Barlat Yld2004-18p'
            trace.line.update(line_styles['Barlat_Yld2004_18p'])
        elif 'Hill1979' in trace.name:
            new_name = 'Hill 1979'
        elif 'Hill1948' in trace.name:
            new_name = 'Hill 1948'
            trace.line.update(line_styles['Hill1948'])
        else:
            new_name = trace.name

        trace.name = new_name
        trace.line.update({
            'color': DEFAULT_PLOTLY_COLORS[idx],
            'width': 1,
        })

    fig_wig.layout.update({
        'width': 280,
        'height': 280,
        'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
        'template': 'simple_white',
        'xaxis': {
            'mirror': 'ticks',
            'ticks': 'inside',
            'range': [0.1, 1.2],
            'dtick': 0.2,
            'title': r'\yldFuncFigXLab{}',
            'tickformat': '.1f',
        },
        'yaxis': {
            'mirror': 'ticks',
            'ticks': 'inside',
            'range': [0.1, 1.2],
            'dtick': 0.2,
            'title': r'\yldFuncFigYLab{}',
            'tickformat': '.1f',
        },
        'legend': {
            'x': 0.07,
            'y': 0.1,
            'xanchor': 'left',
            'yanchor': 'bottom',
            'bgcolor': 'rgba(255, 255, 255, 0)',
            'tracegroupgap': 0,
        }
    })
    return fig_wig


def compare_yield_function_yield_points(load_response_set, ypc_slice, **kwargs):
    yld_funcs_subset = []
    yld_points = []
    for i in load_response_set.fitted_yield_functions[ypc_slice]:
        yld_funcs_subset.append(i['yield_function'])
        yld_points.append(
            load_response_set.yield_point_criteria[i['YPC_idx']].values[0, i['YPC_value_idx']])

    return YieldFunction.compare_2D(yld_funcs_subset, legend_text=yld_points, **kwargs)


def compare_yield_function_types(all_load_responses, yield_function_idx=0, **kwargs):
    yld_funcs_subset = [
        resp.fitted_yield_functions[yield_function_idx]['yield_function']
        for resp in all_load_responses
    ]

    return YieldFunction.compare_2D(yld_funcs_subset, **kwargs)


def show_all_fitted_yield_function_parameters(all_load_responses):
    all_fitted_params = {}
    for yld_func_type_idx in all_load_responses:
        yld_func_type_fitted_params = []
        for i in yld_func_type_idx.fitted_yield_functions:
            yld_point = all_load_responses[1].yield_point_criteria[i['YPC_idx']
                                                                   ].values[0, i['YPC_value_idx']]
            yld_func_type_fitted_params.append(
                {'yield_point': yld_point, **i['yield_function'].get_parameters()})
        all_fitted_params.update(
            {i['yield_function'].name: yld_func_type_fitted_params})

    tab_children = []
    tab_titles = []
    for yld_func_name, fitted_vals in all_fitted_params.items():
        df = pd.DataFrame(fitted_vals)
        out_widget = widgets.Output()
        tab_children.append(out_widget)
        tab_titles.append(yld_func_name)
        with out_widget:
            display(df)

    tab = widgets.Tab()
    tab.children = tab_children
    for i in range(len(tab_titles)):
        tab.set_title(i, tab_titles[i])
    display(tab)
    return all_fitted_params


def collect_hardening_data(sim_tasks, yield_stress, extrapolations=None,
                           plastic_table_strain_interval=2e-3):

    if not extrapolations:
        extrapolations = [{'type': None}]

    hardening_data = {}
    for strain_path, sim_task in sim_tasks.items():
        vol_elem_response = sim_task.elements[0].outputs.volume_element_response
        strain_vM_total = vol_elem_response['vol_avg_equivalent_strain']['data']
        strain_vM_plastic = vol_elem_response['vol_avg_equivalent_plastic_strain']['data']
        stress_vM = vol_elem_response['vol_avg_equivalent_stress']['data']

        # Smooth oscillations in stress for higher strains:
        smooth_idx = np.where(strain_vM_total > 0.15)[0][0]
        lowess_out = lowess(stress_vM, strain_vM_total,
                            is_sorted=True, frac=0.025, it=0, return_sorted=False)
        stress_vM_smooth = np.concatenate((
            stress_vM[:smooth_idx],
            lowess_out[smooth_idx:]
        ))

        extrapolated_data = []
        for extrap in extrapolations:

            extrap_mode = extrap['type']
            extrap_to_strain = extrap.get('extrapolate_to')

            # For extrapolations that perform a fit, record the fit as well:
            strain_vM_plastic_fit = None
            stress_vM_smooth_fit = None

            strain_vM_plastic_extrap = None
            stress_vM_smooth_extrap = None

            # Extrapolate (strain_vM_plastic, stress_vM_smooth):
            if extrap_mode == 'final_stress':
                add_strain = np.linspace(
                    strain_vM_plastic[-1], extrap_to_strain, 1000)
                add_stress = np.zeros_like(add_strain) + stress_vM_smooth[-1]

            elif extrap_mode == 'final_work_hardening_rate':
                # Continue final work hardening rate
                linear_fit_num = extrap['linear_fit_num']
                (linear_m, linear_c), pcov = curve_fit(
                    linear_model,
                    strain_vM_plastic[-linear_fit_num:],
                    stress_vM_smooth[-linear_fit_num:],
                )
                add_strain = np.linspace(
                    strain_vM_plastic[-1], extrap_to_strain, 1000)
                add_stress = linear_model(add_strain, linear_m, linear_c)

                # Separately get the fitted data:
                strain_vM_plastic_fit = strain_vM_plastic[-linear_fit_num:]
                stress_vM_smooth_fit = linear_model(
                    strain_vM_plastic[-linear_fit_num:], linear_m, linear_c)

            elif extrap_mode == 'power_law':
                # Fit the stress-strain data to a power law and extrapolate from that:

                # Option to fit for strains greater than some value:
                fit_idx = strain_vM_plastic > 0.01
                strain_vM_plastic_fit = strain_vM_plastic[fit_idx]

                (power_law_K, power_law_exp), pcov = curve_fit(
                    hardening_power_law,
                    strain_vM_plastic_fit,
                    stress_vM_smooth[fit_idx],
                    p0=(100e6, 1),
                )
                # Separately get the fitted data:
                stress_vM_smooth_fit = hardening_power_law(
                    strain_vM_plastic_fit, power_law_K, power_law_exp)

                if extrap.get('smooth_transition', True):
                    # For a smooth transition, take the fit back to the point at which it is within
                    # some threshold of the input data:

                    OVERLAP_THRESHOLD = 0.01e6
                    overlap_idx_fit = np.where(
                        np.abs(
                            (stress_vM_smooth[-stress_vM_smooth_fit.size:] -
                            stress_vM_smooth_fit)
                        ) < OVERLAP_THRESHOLD
                    )[0][-1]
                    overlap_idx = np.where(fit_idx)[0][0] + overlap_idx_fit

                    truncated_percent = (strain_vM_plastic.size -
                                        overlap_idx) * 100 / strain_vM_plastic.size
                    print(f'Power law extrapolation: truncating {truncated_percent:.0f}% of input data '
                        f'to ensure smooth transition to extrapolated data.')

                    new_strain = np.linspace(strain_vM_plastic[overlap_idx], extrap_to_strain, 1000)
                    new_stress = hardening_power_law(new_strain, power_law_K, power_law_exp)

                    strain_vM_plastic_extrap = np.concatenate(
                        (strain_vM_plastic[:overlap_idx], new_strain[1:]))
                    stress_vM_smooth_extrap = np.concatenate(
                        (stress_vM_smooth[:overlap_idx], new_stress[1:]))                        
                else:
                    add_strain = np.linspace(strain_vM_plastic[-1], extrap_to_strain, 1000)
                    add_stress = hardening_power_law(add_strain, power_law_K, power_law_exp)

            elif extrap_mode == 'constant_work_hardening_rate':
                # Assume this number to be the work hardening rate at which to extrapolate
                known_point = (strain_vM_plastic[-1], stress_vM_smooth[-1])
                linear_m = extrap['work_hardening_rate']
                linear_c = known_point[1] - (linear_m * known_point[0])
                add_strain = np.linspace(
                    strain_vM_plastic[-1], extrap_to_strain, 1000)
                add_stress = linear_model(add_strain, linear_m, linear_c)

            elif extrap_mode is None:
                # No extrapolation
                strain_vM_plastic_extrap = strain_vM_plastic
                stress_vM_smooth_extrap = stress_vM_smooth

            else:
                raise NotImplementedError(
                    f'Unknown extrapolation mode: {extrap_mode}')

            if strain_vM_plastic_extrap is None:
                strain_vM_plastic_extrap = np.concatenate(
                    (strain_vM_plastic, add_strain[1:]))
                stress_vM_smooth_extrap = np.concatenate(
                    (stress_vM_smooth, add_stress[1:]))

            work_hardening_rate = np.gradient(
                stress_vM_smooth_extrap, strain_vM_plastic_extrap)

            # Skip transition point to smoothed data:
            work_hardening_rate = np.concatenate((
                work_hardening_rate[:(smooth_idx - 1)],
                work_hardening_rate[(smooth_idx + 1):],
            ))

            # Interpolate a subset of (strain_vM_plastic_extrap, stress_vM_smooth_extrap) for FE plastic table:
            if extrap_mode is None:
                extrapolate_to_strain_i = np.max(strain_vM_plastic)
                plastic_table_size = int(0.3 / plastic_table_strain_interval)
            else:
                extrapolate_to_strain_i = extrap_to_strain
                plastic_table_size = int(
                    extrapolate_to_strain_i / plastic_table_strain_interval)

            strain_vM_plastic_extrap_subset = np.linspace(
                0, extrapolate_to_strain_i, num=plastic_table_size)
            stress_vM_smooth_extrap_subset = np.interp(
                strain_vM_plastic_extrap_subset,
                strain_vM_plastic_extrap,
                stress_vM_smooth_extrap,
            )
            # Set the first interpolation value (zero-strain) to the approximate yield stress:
            stress_vM_smooth_extrap_subset[0] = yield_stress

            extrapolated_data.append({
                'strain_vM_plastic_extrap': strain_vM_plastic_extrap,
                'stress_vM_smooth_extrap': stress_vM_smooth_extrap,
                'strain_vM_plastic_fit': strain_vM_plastic_fit,
                'stress_vM_smooth_fit': stress_vM_smooth_fit,
                'work_hardening_rate': work_hardening_rate,
                'strain_vM_plastic_extrap_subset': strain_vM_plastic_extrap_subset,
                'stress_vM_smooth_extrap_subset': stress_vM_smooth_extrap_subset,
                'plastic_table_size': plastic_table_size,
                'extrapolation': extrap,
            })

        hardening_data[strain_path] = {
            'strain_vM_total': strain_vM_total,
            'strain_vM_plastic': strain_vM_plastic,
            'stress_vM': stress_vM,
            'stress_vM_smooth': stress_vM_smooth,
            'yield_stress': yield_stress,
            'plastic_table_strain_interval': plastic_table_strain_interval,
            'extrapolated_data': extrapolated_data,
        }

    return hardening_data


def get_extrapolation_mode_plot_data(strain_path, strain_path_idx, hard_data,
                                     extrapolation_type, legend_names, show_interpolation,
                                     show_non_extrapolated_stress, show_fitted_data):

    hard_data_extrap = [i for i in hard_data['extrapolated_data']
                 if i['extrapolation']['type'] == extrapolation_type][0]

    legend_data = {
        'name': STRAIN_PATH_LEGEND_NAMES[strain_path],
        'legendgroup': STRAIN_PATH_LEGEND_NAMES[strain_path],
    }
    plt_data = []
    if show_interpolation:
        # Show the subset of data used to generate the FE plastic tables:
        plt_data.append({
            'x': hard_data_extrap['strain_vM_plastic_extrap_subset'],
            'y': hard_data_extrap['stress_vM_smooth_extrap_subset'] / 1e6,
            'xaxis': 'x1',
            'yaxis': 'y1',
            'text': np.arange(hard_data_extrap['strain_vM_plastic_extrap'].size),
            'mode': 'markers',
            'marker': {
                'color': qualitative.D3[strain_path_idx],
                'symbol': 'circle',
                'size': 4,
            },
            'showlegend': False,
            **legend_data,
        })

    plt_data.extend([
        {
            'x': hard_data_extrap['strain_vM_plastic_extrap'],
            'y': hard_data_extrap['stress_vM_smooth_extrap'] / 1e6,
            'xaxis': 'x1',
            'yaxis': 'y1',
            'text': np.arange(hard_data_extrap['strain_vM_plastic_extrap'].size),
            'line': {
                'width': 1,
                'color': qualitative.D3[strain_path_idx],
            },
            'showlegend': True,
            **legend_data,
        },
        {
            'x': hard_data_extrap['strain_vM_plastic_extrap'],
            'y': hard_data_extrap['work_hardening_rate'] / 1e9,
            'xaxis': 'x1',
            'yaxis': 'y2',
            'text': np.arange(hard_data_extrap['strain_vM_plastic_extrap'].size),
            'line': {
                'width': 1,
                'color': qualitative.D3[strain_path_idx],
            },
            'showlegend': False,
            **legend_data,
        },
    ])

    if show_non_extrapolated_stress:
        plt_data.append({
            'x': hard_data['strain_vM_plastic'],
            'y': hard_data['stress_vM_smooth'] / 1e6,
            'xaxis': 'x1',
            'yaxis': 'y1',
            'text': np.arange(hard_data['strain_vM_plastic'].size),
            'line': {
                'width': 2,
                'color': qualitative.D3[strain_path_idx],
            },
            'showlegend': False,
            **legend_data,
        })

    if show_fitted_data:
        # If the extrapolation required a fit, show the fit data as well:
        x = hard_data_extrap.get('strain_vM_plastic_fit', [])
        y = hard_data_extrap.get('stress_vM_smooth_fit')        
        plt_data.append({
            'x': x,
            'y': y / 1e6 if y is not None else [],
            'xaxis': 'x1',
            'yaxis': 'y1',
            'text': np.arange(hard_data_extrap['strain_vM_plastic_fit'].size) if y is not None else [],
            'line': {
                'dash': 'dash',
                'color': qualitative.D3[strain_path_idx],
                'width': 2,
            },
            'showlegend': False,
            **legend_data,
        })

    return plt_data


def explore_extrapolated_work_hardening(hardening_data, extrapolation_type=None, show_interpolation=True,
                                        show_non_extrapolated_stress=False, show_fitted_data=None,
                                        layout_args=None):
    """Plot Von Mises stress-plastic-strain extrapolated data and work hardening rate curves from simulations."""

    layout_args = layout_args or {}

    plt_data = []
    for strain_path_idx, (strain_path, hard_data) in enumerate(hardening_data.items()):
        plt_data.extend(
            get_extrapolation_mode_plot_data(
                strain_path,
                strain_path_idx,
                hard_data,
                extrapolation_type,
                STRAIN_PATH_LEGEND_NAMES,
                show_interpolation,
                show_non_extrapolated_stress,
                show_fitted_data,
            )
        )

    fig = graph_objects.FigureWidget(
        data=plt_data,
        layout={
            'template': 'simple_white',
            'xaxis': {
                'mirror': 'ticks',
                'ticks': 'inside',
                'title': {
                    'text': r'Von Mises true plastic strain, \strainVM{{}}',
                    'font': {
                        'size': 12,
                    },
                },
            },
            'yaxis': {
                'anchor': 'x',
                'ticks': 'inside',
                'title': {
                    'text': r'Von Mises true stress, \stressVM{} (\MPa{})',
                    'font': {
                        'size': 12,
                    },
                },
            },
            'yaxis2': {
                'anchor': 'x',
                'side': 'right',
                'overlaying': 'y',
                'range': [0, 2.5],
                'title': {
                    'text': r'Work hardening rate (\GPa{})',
                    'font': {
                        'size': 12,
                    },
                }
            },
            'legend': {
                'x': 0.80,
                'y': 0.5,
                'xanchor': 'right',
                'yanchor': 'middle',
                'tracegroupgap': 0,
            },
            **layout_args,
        }
    )
    toggle_opts = [i['extrapolation']['type']
                   for i in hardening_data['uniaxial']['extrapolated_data']]
    toggle_buttons = widgets.ToggleButtons(
        description='Extrapolation: ',
        options=toggle_opts,
    )
    explorer = WorkHardeningExplorer(
        hardening_data=hardening_data,
        figure=fig,
        buttons=toggle_buttons,
        show_interpolation=show_interpolation,
        show_non_extrapolated_stress=show_non_extrapolated_stress,
        show_fitted_data=show_fitted_data,
        layout_args=layout_args
    )

    return explorer.visual


def show_work_hardening(hardening_data, layout_args=None):
    """Plot Von Mises stress-strain data and work hardening rate curves from simulations."""

    layout_args = layout_args or {}
    plt_data = []
    for strain_path_idx, (strain_path, hard_data) in enumerate(hardening_data.items()):

        zero_extrap_data_idx = [idx for idx, i in enumerate(hard_data['extrapolated_data'])
                                if i['extrapolation']['type'] is None][0]
        zero_extrap_data = hard_data['extrapolated_data'][zero_extrap_data_idx]

        legend_data = {
            'name': STRAIN_PATH_LEGEND_NAMES[strain_path],
            'legendgroup': STRAIN_PATH_LEGEND_NAMES[strain_path],
        }
        plt_data.extend([
            {
                'x': zero_extrap_data['strain_vM_plastic_extrap'],
                'y': zero_extrap_data['stress_vM_smooth_extrap'] / 1e6,
                'xaxis': 'x1',
                'yaxis': 'y1',
                'text': np.arange(zero_extrap_data['strain_vM_plastic_extrap'].size),
                'line': {
                    'width': 1,
                    'color': qualitative.D3[strain_path_idx],
                },
                'showlegend': True,
                **legend_data,
            },
            {
                'x': hard_data['strain_vM_total'][:-3],
                'y': zero_extrap_data['work_hardening_rate'][:hard_data['strain_vM_total'].size - 3] / 1e9,
                'xaxis': 'x1',
                'yaxis': 'y2',
                'text': np.arange(zero_extrap_data['strain_vM_plastic_extrap'].size),
                'line': {
                    'width': 1,
                    'color': qualitative.D3[strain_path_idx],
                    'dash': 'dot',
                },
                'showlegend': False,
                **legend_data,
            },
        ])

    layout = {
        'width': 400,
        'height': 350,
        'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
        'template': 'simple_white',
        'xaxis': {
            'range': [-0.01, 0.3],
            'mirror': 'ticks',
            'ticks': 'inside',
            'dtick': 0.1,
            'tickformat': '.1f',
            'title': {
                'text': r'Von Mises true strain, \strainVM{{}}',
                'font': {
                    'size': 12,
                },
            },

        },
        'yaxis': {
            'anchor': 'x',
            'ticks': 'inside',
            'range': [0, 330],
            'title': {
                'text': r'Von Mises true stress, \stressVM{} (\MPa{})',
                'font': {
                    'size': 12,
                },
            },
        },
        'yaxis2': {
            'anchor': 'x',
            'side': 'right',
            'overlaying': 'y',
            'tickformat': '.1f',
            'title': {
                'text': r'Work hardening rate (\GPa{})',
                'font': {
                    'size': 12,
                },
            },
            'range': [0, 2.5],
            'ticks': 'inside',
        },
        'legend': {
            'x': 0.80,
            'y': 0.5,
            'xanchor': 'right',
            'yanchor': 'middle',
            'tracegroupgap': 0,
        },
        **layout_args,
    }
    fig = graph_objects.FigureWidget(
        data=plt_data,
        layout=layout
    )

    return fig


def prepare_plastic_tables(hardening_data):
    """Generate some Dataframes of the plastic tables."""

    # First extract out extrapolation types:
    extrap_types = [i['extrapolation']['type']
                    for i in hardening_data['uniaxial']['extrapolated_data']]
    all_dataframes = {}

    for extrap_type in extrap_types:

        column_headers = []
        interpolated_hardening_data = []
        for k, v in hardening_data.items():
            hard_data = [i for i in v['extrapolated_data']
                         if i['extrapolation']['type'] == extrap_type][0]

            interpolated_hardening_data.append(
                hard_data['strain_vM_plastic_extrap_subset'])
            interpolated_hardening_data.append(
                hard_data['stress_vM_smooth_extrap_subset'])
            column_headers.extend([(k, 'strain'), (k, 'stress')])

        df = pd.DataFrame(
            data=np.array(interpolated_hardening_data).T,
            columns=pd.MultiIndex.from_tuples(column_headers)
        )
        all_dataframes.update({extrap_type: df})

    return all_dataframes


def show_plastic_stress_strain_data(hardening_data, extrapolation_type):
    interpolated_hardening_data = []
    column_headers = []
    for k, v in hardening_data.items():

        hard_data = [i for i in v['extrapolated_data']
                     if i['extrapolation']['type'] == extrapolation_type][0]

        interpolated_hardening_data.append(
            hard_data['strain_vM_plastic_extrap_subset'])
        interpolated_hardening_data.append(
            hard_data['stress_vM_smooth_extrap_subset'])
        column_headers.extend([(k, 'strain'), (k, 'stress')])

    interpolated_hardening_data = np.array(interpolated_hardening_data)
    df = pd.DataFrame(
        data=interpolated_hardening_data.T,
        columns=pd.MultiIndex.from_tuples(column_headers)
    )
    display(df)
    return df


def get_latex_yield_func_params(yield_func_name, yield_func_param_vals, val_format='.4f', pad_to=18):
    """Format yield function parameter symbols and values, given some macros defined
    in the manuscript.
    """

    vals = []
    latex_keys = []
    for k, v in yield_func_param_vals.items():

        if k in ['yield_point', 'equivalent_stress']:
            continue

        vals.append(f'{v:{val_format}}')

        if yield_func_name == 'Barlat_Yld2004_18p':
            if '_p_' in k:
                new_k = f'\\yldFuncLinTransComp{{\\myprime}}{{{k[-2:]}}}'
            elif '_dp_' in k:
                new_k = f'\\yldFuncLinTransComp{{\\mydprime}}{{{k[-2:]}}}'
            elif k == 'exponent':
                new_k = '\\yldFuncExp{}'
            else:
                raise ValueError(k)
            latex_keys.append(new_k)

        elif yield_func_name == 'Barlat_Yld91':
            if k == 'exponent':
                new_k = '\\yldFuncExp{}'
            else:
                new_k = f'${k.upper()}$'
            latex_keys.append(new_k)

        elif yield_func_name == 'Hill1948':
            if k == 'exponent':
                continue
            new_k = '$' + k + '_\\mathrm{h}$'
            latex_keys.append(new_k)

        else:
            raise ValueError(f'Unknown yield_func_name: {yield_func_name}')

    if len(latex_keys) < pad_to:
        latex_keys += [''] * (pad_to - len(latex_keys))
        vals += ['-'] * (pad_to - len(vals))

    return np.array([latex_keys, vals])


def show_FLC(FLC):
    plt_data = []
    for strain_path_i in FLC['strain_paths']:
        name = f'BCs: {strain_path_i["displacement_BCs"]}; Groove angle: {strain_path_i["groove_angle_deg"]}'
        if strain_path_i['strain'] is None:
            continue
        plt_data.append({
            'x': strain_path_i['strain'][0],
            'y': strain_path_i['strain'][1],
            'mode': 'markers',
            'name': name,
        })

    plt_data.append({
        'x': FLC['forming_limits'][0],
        'y': FLC['forming_limits'][1],
        'mode': 'lines',
        'name': 'Forming limit',
    })

    fig_wig = graph_objects.FigureWidget(
        data=plt_data,
        layout={
            'width': 800,
            'xaxis': {
                'range': [-0.5, 0.5],
                'title': 'Minor strain',
            },
            'yaxis': {
                'range': [0, 0.5],
                'title': 'Major strain',
                'scaleanchor': 'x',
            }
        },
    )
    return fig_wig


def plot_FLD_full(simulated_FLC, necking_strains, fracture_strains, constellium_data):
    FLD_full_fig = graph_objects.FigureWidget(
        data=[
            {
                'x': constellium_data[:, 0],
                'y': constellium_data[:, 1],
                'name': '*Surf.',
                'mode': 'lines',
                'line': {
                    'width': 1.5,
                    'color': qualitative.D3[3],
                },
            },
            {
                'x': constellium_data[:, 2],
                'y': constellium_data[:, 3],
                'name': '*Surf. HF',
                'mode': 'lines',
                'line': {
                    'width': 1.5,
                    'color': qualitative.D3[2],
                },
            },
            {
                'x': fracture_strains[:, 0],
                'y': fracture_strains[:, 1],
                'name': 'Frac. strain       ',  # (spaces for legend padding!)
                'mode': 'markers',
                'marker': {
                    'size': 8,
                    'symbol': 'circle-open',
                    'color': qualitative.D3[0],
                },
            },
            {
                'x': necking_strains[:, 0],
                'y': necking_strains[:, 1],
                'name': 'Neck. strain',
                'mode': 'markers',
                'marker': {
                    'size': 8,
                    'symbol': 'cross',
                    'color': qualitative.D3[1],
                },
            },
            {
                'x': simulated_FLC['forming_limits'][0],
                'y': simulated_FLC['forming_limits'][1],
                'mode': 'markers+lines',
                'name': 'Simulated',
                'marker': {
                    'size': 7,
                },
                'line': {
                    'width': 1.2,
                    'color': qualitative.D3[4],
                },
            },

        ],
        layout={
            'width': 350,
            'height': 350,
            'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
            'template': 'simple_white',
            'xaxis': {
                'title': 'Minor strain',
                'mirror': 'ticks',
                'ticks': 'inside',
                'dtick': 0.1,
                'tickformat': '.1f',
                'showgrid': True,
            },
            'yaxis': {
                'range': [0.09, 0.59],
                'scaleanchor': 'x',
                'title': 'Major strain',
                'mirror': 'ticks',
                'ticks': 'inside',
                'dtick': 0.1,
                'tickformat': '.1f',
                'showgrid': True,
            },
            'legend': {
                'x': 0.98,
                'y': 0.98,
                'xanchor': 'right',
                'yanchor': 'top',
                'tracegroupgap': 0,
                'bgcolor': 'rgb(250, 250, 250)',
            },
        }
    )
    return FLD_full_fig


def plot_strain_paths_to_necking_plotly(strain_at_necking, sample_sizes):
    plt_data = []
    for exp_name, strain in strain_at_necking.items():
        color_idx = sample_sizes.index(exp_name.split("_")[0])
        color = qualitative.D3[color_idx]
        plt_data.append({
            'x': strain[:, 0],
            'y': strain[:, 1],
            'name': exp_name,
            'mode': 'lines',
            'line': {
                'width': 1.2,
                'color': color,
                'dash': ('solid', 'dash', 'dot')[int(exp_name.split("_")[1]) - 1],
            },
            'showlegend': False,
            'legendgroup': exp_name.split("_")[0],

        })
        plt_data.append({
            'x': [strain[-1, 0]],
            'y': [strain[-1, 1]],
            'name': exp_name,
            'legendgroup': exp_name.split("_")[0],
            'mode': 'markers',
            'marker': {
                'size': 10,
                'color': color,
            },
            'showlegend': False,
        })
    annots = [
        {
            'x': -0.065,
            'y': 0.125,
            'text': r'\formLimsSubRefA{}',
            'showarrow': False,
            'font': {
                'size': 4,
            },
        },
        {
            'x': -0.065,
            'y': 0.29,
            'text': r'\formLimsSubRefB{}',
            'showarrow': False,
            'font': {
                'size': 4,
            },
        },
        {
            'x': -0.045,
            'y': 0.29,
            'text': r'\formLimsSubRefC{}',
            'showarrow': False,
            'font': {
                'size': 4,
            },
        },
        {
            'x': -0.018,
            'y': 0.29,
            'text': r'\formLimsSubRefD{}',
            'showarrow': False,
            'font': {
                'size': 4,
            },
        },
        {
            'x': 0.018,
            'y': 0.29,
            'text': r'\formLimsSubRefE{}',
            'showarrow': False,
            'font': {
                'size': 4,
            },
        },
        {
            'x': 0.09,
            'y': 0.29,
            'text': r'\formLimsSubRefF{}',
            'showarrow': False,
            'font': {
                'size': 4,
            },
        },
    ]
    fig = graph_objects.FigureWidget(
        data=plt_data,
        layout={
            'width': 350,
            'height': 310,
            'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
            'template': 'simple_white',
            'xaxis': {
                'range': [-0.08, 0.105],
                'title': r'Minor strain, \minorStrain{}',
                'mirror': 'ticks',
                'ticks': 'inside',
                'tickformat': '.2f',
            },
            'yaxis': {
                'range': [0.0, 0.32],
                'title': r'Major strain, \majorStrain{}',
                'mirror': 'ticks',
                'ticks': 'inside',
                'tickformat': '.2f',
            },
            'annotations': annots,
        },
    )
    return fig


def linear_model(x, m, c):
    return m*x + c


def hardening_power_law(plastic_strain, K, exponent):
    return K * (plastic_strain ** exponent)


def get_yield_function_fitting_error(all_load_responses, yield_function_idx):

    error_data = {}
    for idx, load_resp in enumerate(all_load_responses):
        abs_residual = np.abs(
            load_resp.fitted_yield_functions[yield_function_idx]['yield_function'].fit_info.fun)
        name = load_resp.fitted_yield_functions[yield_function_idx]['yield_function'].name
        error_data.update({name: abs_residual})

    return error_data


def show_yield_function_fitting_error(all_load_responses, yield_function_idx, layout_args=None):
    """Plot the yield function fitting residual at the optimised solution."""

    legend_names = {
        'Hill1948': 'Hill 1948',
        'Barlat_Yld91': 'Bar.\ Yld91',
        'Barlat_Yld2004_18p': 'Bar.\ Yld2004-18p',
    }

    plt_data = []
    err_data = get_yield_function_fitting_error(
        all_load_responses, yield_function_idx)
    for idx, (name, err_data) in enumerate(err_data.items()):
        load_resp = all_load_responses[idx]
        plt_data.append({
            'type': 'histogram',
            'x': err_data * 1e2,
            'xbins': {
                'size': 0.01 * 1e2
            },
            'marker': {
                'color': qualitative.D3[idx],
            },
            'name': legend_names.get(name, name),
        })

    layout = {
        'template': 'simple_white',
        'width': 280,
        'height': 280,
        'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
        'xaxis': {
            'title': {
                'text': r'\yldFuncResidualXLab{}',
            },
            'dtick': 0.01 * 1e2,
            'mirror': 'ticks',
            'ticks': 'inside',
        },
        'yaxis': {
            'title': {
                'text': 'Number of stress states',
            },
            'mirror': 'ticks',
            'ticks': 'inside',
        },
        'legend': {
            'x': 0.3,
            'y': 0.9,
            'xanchor': 'left',
            'yanchor': 'top',
            'bgcolor': 'rgba(255, 255, 255, 0)',
            'tracegroupgap': 0,
        }
    }
    fig = graph_objects.FigureWidget(
        data=plt_data,
        layout={**layout, **(layout_args or {})}
    )
    return fig


def plot_yield_function_exponent_evolution(all_fitted_params):
    barlat_18p_exp_evo = [i['exponent']
                          for i in all_fitted_params['Barlat_Yld2004_18p']]
    barlat_6p_exp_evo = [i['exponent']
                         for i in all_fitted_params['Barlat_Yld91']]
    yield_points = [i['yield_point']
                    for i in all_fitted_params['Barlat_Yld2004_18p']]
    fig = graph_objects.FigureWidget(
        data=[
            {
                'x': yield_points,
                'y': barlat_6p_exp_evo,
                'name': 'Barlat Yld91',
                'line': {
                    'color': qualitative.D3[1],
                },
            },
            {
                'x': yield_points,
                'y': barlat_18p_exp_evo,
                'name': 'Barlat Yld2004-18p',
                'line': {
                    'color': qualitative.D3[2],
                },
            },
        ],
        layout={
            'xaxis_title': 'Yield point (Von Mises plastic strain)',
            'yaxis_title': 'Yield function exponent, m',
        }
    )
    return fig


def get_strain_ratios(vol_avg_def_grad, sheet_dirs=None):
    """Get the RVE strains and the Lankford coefficient from the volume-averaged
    deformation gradient, using the diagonal components of the Green strain.

    Parameters
    ----------
    vol_avg_def_grad : ndarray of shape (N, 3, 3)
        Deformation gradient tensor for each of N simulation increments.
    sheet_dirs : dict of (str: str), optional
        Dict assigning each Cartesian direction to RD/TD/ND. If None,
        "x" will be assigned to "RD", "y" to "TD" and "z" to "ND".

    """

    if not sheet_dirs:
        sheet_dirs = {
            'x': 'RD',
            'y': 'TD',
            'z': 'ND',
        }

    sheet_dirs_inv = {v: k for k, v in sheet_dirs.items()}
    cart_dirs = ['x', 'y', 'z']
    sheet_dir_idx = {i: cart_dirs.index(
        sheet_dirs_inv[i]) for i in sheet_dirs.values()}

    F = vol_avg_def_grad
    F_T = np.transpose(F, (0, 2, 1))
    green_strain = 0.5 * ((F_T @ F) - np.eye(3))

    strain_thick = green_strain[:, sheet_dir_idx['ND'], sheet_dir_idx['ND']]
    strain_trans = green_strain[:, sheet_dir_idx['TD'], sheet_dir_idx['TD']]
    strain_long = green_strain[:, sheet_dir_idx['RD'], sheet_dir_idx['RD']]

    true_strain_trans = np.log(strain_trans[1:] + 1)
    true_strain_long = np.log(strain_long[1:] + 1)
    true_strain_thick = np.log(strain_thick[1:] + 1)
    true_strain_vol = true_strain_trans + true_strain_long + true_strain_thick

    lankford = true_strain_trans / -(true_strain_long + true_strain_trans)
    # lankford = true_strain_trans / true_strain_thick

    out = {
        'strain_thick': strain_thick,
        'strain_trans': strain_trans,
        'strain_long': strain_long,
        'strain_vol': true_strain_vol,
        'lankford': lankford,
    }
    return out


def get_simulated_lankford_parameter(workflow):

    sim_elem = workflow.tasks.simulate_volume_element_loading.elements[0]
    vol_avg_def_grad = sim_elem.outputs.volume_element_response['vol_avg_def_grad']['data']
    true_strain = sim_elem.outputs.volume_element_response['vol_avg_equivalent_strain']['data']
    lankford = get_strain_ratios(vol_avg_def_grad)['lankford']

    return (true_strain, lankford)


def plot_lankford_parameter_comparison(lankford_parameter_evolution):

    plt_data = [
        {
            'x': true_strain,
            'y': R,
            'name': name,
            'line': {
                'color': 'black' if 'Random' in name else ('blue' if 'Surfalex' in name else (qualitative.D3[idx // 2])),
                'dash': ('dot' if idx % 2 == 0 else 'solid') if 'Simulated' not in name else 'dashdot',
                'width': 1.2,
            },
        }
        for idx, (name, (true_strain, R)) in enumerate(lankford_parameter_evolution.items())
    ]

    fig = graph_objects.FigureWidget(
        data=plt_data,
        layout={
            'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
            'template': 'simple_white',
            'xaxis': {
                'title': 'True strain',
                'mirror': 'ticks',
                'ticks': 'inside',
                'range': [0.02, 0.26],
            },
            'yaxis': {
                'title': 'Lankford parameter',
                'mirror': 'ticks',
                'ticks': 'inside',
                'range': [0.1, 1.0],
            },
        },
    )
    return fig


def plot_static_figure_stress_strain_curves(cropped_voltage_data):
    fig = graph_objects.FigureWidget(
        data=[
            {
                'x': data[:, 0],
                'y': data[:, 1],
                'name': f'\\ang{{{exp_name.split("-")[0]}}}',
                'line': {
                    'color': qualitative.D3[idx // 2]
                },
            }
            for idx, (exp_name, data) in enumerate(cropped_voltage_data.items())
            if exp_name[-1] == '1'
        ],
        layout={
            'width': 280,
            'height': 250,
            'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
            'template': 'simple_white',
            'xaxis': {
                'title': 'True strain, \strain{}',
                'mirror': 'ticks',
                'ticks': 'inside',
                'range': [-0.01, 0.35],
                'dtick': 0.10,
                'tickformat': '.1f',
            },
            'yaxis': {
                'title': 'True stress, \stress{} (\MPa)',
                'mirror': 'ticks',
                'ticks': 'inside',
                'range': [0.0, 330],
                'dtick': 50,
            },
            'legend': {
                'x': 1.03,
                'y': 0.10,
                'xanchor': 'right',
                'yanchor': 'bottom',
                'tracegroupgap': 0,
                'bgcolor': 'rgba(255, 255, 255, 0)',
                'font': {
                    'size': 10,
                }
            },
        },
    )
    return fig


def visualise_MK_forming_limits(FLC_workflows, show_3D=False, additional_FLCs=None, layout_2D=None,
                                show_strain_paths=True):

    plt_data_3D = []
    plt_data_2D = []
    major_range = [0, 0]
    minor_range = [0, 0]
    range_margin = 1.1
    for FLC_data_idx, FLC_data in enumerate(FLC_workflows):
        FLC_dict = FLC_data['workflow'].tasks.find_forming_limit_curve.elements[0].outputs.forming_limit_curve
        name = FLC_data['label']
        for strain_path_idx, strain_path in enumerate(FLC_dict['strain_paths']):
            if strain_path['strain'] is not None:

                if min(strain_path['strain'][1]) < major_range[0]:
                    major_range[0] = min(
                        strain_path['strain'][1] * range_margin)

                if max(strain_path['strain'][1]) > major_range[1]:
                    major_range[1] = max(
                        strain_path['strain'][1] * range_margin)

                if min(strain_path['strain'][0]) < minor_range[0]:
                    minor_range[0] = min(
                        strain_path['strain'][0] * range_margin)

                if max(strain_path['strain'][0]) > minor_range[1]:
                    minor_range[1] = max(
                        strain_path['strain'][0] * range_margin)

                plt_data_3D.append({
                    'type': 'scatter3d',
                    'x': strain_path['strain'][0],
                    'y': [strain_path['groove_angle_deg']] * strain_path['strain'].shape[1],
                    'z': strain_path['strain'][1],
                    'name': name,
                    'legendgroup': name,
                    'marker': {
                        'size': 2,
                    },
                    'line': {
                        'color': qualitative.D3[FLC_data_idx % len(qualitative.D3)],
                        'width': 0.5,
                    },
                    'showlegend': (True if strain_path_idx == 0 else False) if show_strain_paths else False,
                    'meta': {
                        'type': 'strain_path',
                        'strain_path_idx': strain_path_idx,
                    },
                    'visible': show_strain_paths,
                })
                plt_data_2D.append({
                    'x': strain_path['strain'][0],
                    'y': strain_path['strain'][1],
                    'name': name,
                    'legendgroup': name,
                    'marker': {
                        'size': 2,
                    },
                    'line': {
                        'color': qualitative.D3[FLC_data_idx % len(qualitative.D3)],
                        'width': 0.5,
                    },
                    'showlegend': (True if strain_path_idx == 0 else False) if show_strain_paths else False,
                    'meta': {
                        'type': 'strain_path',
                        'strain_path_idx': strain_path_idx,
                    },
                    'visible': show_strain_paths,
                })

        # Add final forming limit:
        plt_data_3D.append({
            'type': 'scatter3d',
            'x': FLC_dict['forming_limits'][0],
            'y': FLC_dict['forming_limit_groove_angles_deg'],
            'z': FLC_dict['forming_limits'][1],
            'line': {
                'color': qualitative.D3[FLC_data_idx % len(qualitative.D3)],
            },
            'name': name,
            'legendgroup': name,
            'showlegend': not show_strain_paths,
            'visible': True,
            'meta': {
                'type': 'forming_limit',
            },
        })
        line_style = FLC_data.get('line', {})
        marker_style = FLC_data.get('marker', {})
        plt_data_2D.append({
            'x': FLC_dict['forming_limits'][0],
            'y': FLC_dict['forming_limits'][1],
            'line': {
                'color': qualitative.D3[FLC_data_idx % len(qualitative.D3)],
                **line_style,
            },
            'marker': marker_style,
            'name': name,
            'legendgroup': name,
            'showlegend': not show_strain_paths,
            'visible': True,
            'meta': {
                'type': 'forming_limit',
            },
        })

    for idx, add_FLC in enumerate(additional_FLCs or []):
        plt_data_2D.append({
            'x': add_FLC['minor_strain'],
            'y': add_FLC['major_strain'],
            'meta': {'type': 'additional_FLC'},
            'name': add_FLC['name'],
            'showlegend': True,
            'visible': True,
            'legendgroup': add_FLC['name'],
            'mode': add_FLC.get('mode', 'markers'),
            'marker': add_FLC.get('marker', {}),
            'line': {
                'color': qualitative.D3[(len(FLC_workflows) + idx) % len(qualitative.D3)],
            },
        })
        plt_data_3D.append({
            'type': 'scatter3d',
            'x': add_FLC['minor_strain'],
            'y': [0] * len(add_FLC['minor_strain']),
            'z': add_FLC['major_strain'],
            'meta': {'type': 'additional_FLC'},
            'name': add_FLC['name'],
            'showlegend': True,
            'visible': True,
            'legendgroup': add_FLC['name'],
            'mode': add_FLC.get('mode', 'markers'),
            'marker': add_FLC.get('marker', {}),
            'line': {
                'color': qualitative.D3[(len(FLC_workflows) + idx) % len(qualitative.D3)],
            },
        })

    layout_3D = {
        'height': 900,
        'scene': {
            'xaxis': {
                'title': 'Minor strain',
            },
            'yaxis': {
                'title': 'Groove angle (deg.)',
                'dtick': 15,
            },
            'zaxis': {
                'title': 'Major strain',
            },
            'camera': {
                'center': {'x': 0, 'y': 0, 'z': 0},
                'eye': {'x': 0.4, 'y': -2.0, 'z': 0.3},
                'projection': {'type': 'orthographic'},
            }
        }
    }
    layout_2D = {
        'height': 800,
        'xaxis': {
            'title': 'Minor strain',
            'range': [i * range_margin for i in minor_range],
        },
        'yaxis': {
            'title': 'Major strain',
            'scaleanchor': 'x',
            'range': [i * range_margin for i in major_range]
        },
        **(layout_2D or {}),
    }

    fig_3D = graph_objects.FigureWidget(data=plt_data_3D, layout=layout_3D)
    fig_2D = graph_objects.FigureWidget(data=plt_data_2D, layout=layout_2D)

    fig = fig_3D if show_3D else fig_2D
    show_forming_limits_toggle_button = widgets.ToggleButton(
        description='Show forming limits', value=True)
    show_strain_paths_toggle_button = widgets.ToggleButton(
        description='Show strain paths', value=show_strain_paths)

    explorer = FormingLimitExplorer(
        figure=fig,
        show_forming_limits_toggle_button=show_forming_limits_toggle_button,
        show_strain_paths_toggle_button=show_strain_paths_toggle_button,
        FLC_workflows=FLC_workflows,
    )

    return explorer


def identify_strain_path(displacement_BCs):
    # Assuming +/- displacement BCs are symmetric along a given axis
    # [x, x, y, y]

    strain_path = ''
    if displacement_BCs[2] == 'free' and displacement_BCs[3] == 'free':
        return 'Uniaxial'

    major_BC_abs = abs(displacement_BCs[0]) + abs(displacement_BCs[1])
    minor_BC_abs = abs(displacement_BCs[2]) + abs(displacement_BCs[3])

    if np.isclose(minor_BC_abs, 0):
        strain_path = 'Plane strain'

    elif major_BC_abs == minor_BC_abs:
        strain_path = 'Biaxial: ε_min = ε_maj'

    else:
        strain_path = f'Biaxial: ε_min = {major_BC_abs/minor_BC_abs:.2f} ε_maj'

    return strain_path


def plot_static_figure_full_FLC(FLC_workflows, additional_FLCs):

    layout = {
        'width': 350,
        'height': 350,
        'margin': {'t': 20, 'b': 20, 'l': 20, 'r': 20},
        'template': 'simple_white',
        'xaxis': {
            'mirror': 'ticks',
            'ticks': 'inside',
            'range': [-0.2, 0.5],
            'title': r'Minor strain, \minorStrain{}',
            'dtick': 0.1,
            'tickformat': '.1f',
            'showgrid': True,
        },
        'yaxis': {
            'mirror': 'ticks',
            'ticks': 'inside',
            'range': [0.05, 0.75],
            'title': r'Major strain, \majorStrain{}',
            'dtick': 0.1,
            'tickformat': '.1f',
            'showgrid': True,
            'scaleanchor': 'x',
        },
        'legend': {
            'x': 0.98,
            'xanchor': 'right',
            'y': 0.98,
            'yanchor': 'top',
            'tracegroupgap': 0,
            'bgcolor': 'rgb(250, 250, 250)',
        }
    }
    forming_limit_explorer = visualise_MK_forming_limits(
        FLC_workflows,
        show_3D=False,
        additional_FLCs=additional_FLCs,
        layout_2D=layout,
        show_strain_paths=False,
    )

    return forming_limit_explorer.figure
