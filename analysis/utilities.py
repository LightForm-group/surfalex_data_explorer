import hashlib
import pathlib
import urllib.request
import zipfile

import numpy as np
import tqdm
import yaml
import pandas as pd
from plotly import graph_objects
from plotly.colors import DEFAULT_PLOTLY_COLORS
from ipywidgets import widgets
from IPython.display import display
from formable.yielding import YieldFunction


def get_file_from_url(data_folder, url, name, md5=None, unzip=False):
    """Download a file from a URL if it is not already present in `data_folder`. Return
    the local path to the downloaded file. If `unzip=True`, the file is assumed to be a
    zip folder, which will then be extracted to the `data_folder`. In this case, the
    `data_folder` path is returned.

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
        if unzip:
            print('Unzipping...', end='')
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(data_folder)
            print('complete.')

    if unzip:
        out_path = data_folder

    return out_path


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
        all_fitted_params.update({i['yield_function'].name: yld_func_type_fitted_params})

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


def collect_hardening_data(sim_tasks, yield_stress):

    # Collect interpolated stresses to feed to the FE Marciniak-Kuczynski model:
    strain_paths = list(sim_tasks.keys())
    interp_strains = np.linspace(0, 0.29, num=40)

    hardening_data = {k: None for k in strain_paths}
    for strain_path, sim_task in sim_tasks.items():
        strain = sim_task.elements[0].outputs.volume_element_response['vol_avg_equivalent_plastic_strain']['data']
        stress = sim_task.elements[0].outputs.volume_element_response['vol_avg_equivalent_stress']['data']
        hardening_data[strain_path] = {
            'strain': strain,
            'stress': stress,
            'interpolated_strain': interp_strains,
            'interpolated_stress': np.interp(interp_strains, strain, stress),
            'hardening_rate': np.gradient(stress),
        }
        # Set the first interpolation value (zero-strain) to the approximate yield stress:
        hardening_data[strain_path]['interpolated_stress'][0] = yield_stress

    return hardening_data


def show_interpolated_plastic_stress_strain_data(hardening_data):
    interpolated_hardening_data = []
    column_headers = []
    for k, v in hardening_data.items():
        interpolated_hardening_data.append(v['interpolated_strain'])
        interpolated_hardening_data.append(v['interpolated_stress'])
        column_headers.extend([(k, 'interpolated_strain'), (k, 'interpolated_stress')])
    interpolated_hardening_data = np.array(interpolated_hardening_data)
    df = pd.DataFrame(data=interpolated_hardening_data.T,
                      columns=pd.MultiIndex.from_tuples(column_headers))
    display(df)
    return df


def get_latex_yield_func_params(yield_func_name, yield_func_param_vals, val_format='.4f', pad_to=18):
    """Format yield function parameter symbols and values, given some macros defined
    in the manuscript.
    """

    vals = []
    latex_keys = []
    for k, v in yield_func_param_vals.items():

        if k in ['yield_point', 'equivalent_stress', 'exponent']:
            continue

        vals.append(f'{v:{val_format}}')

        if yield_func_name == 'Barlat_Yld2004_18p':
            if '_p_' in k:
                new_k = f'\\yldFuncLinTransComp{{\\myprime}}{{{k[-2:]}}}'
            elif '_dp_' in k:
                new_k = f'\\yldFuncLinTransComp{{\\mydprime}}{{{k[-2:]}}}'
            else:
                raise ValueError(k)
            latex_keys.append(new_k)

        elif yield_func_name == 'Barlat_Yld91':
            new_k = f'${k.upper()}$'
            latex_keys.append(new_k)

        elif yield_func_name == 'Hill1948':
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
