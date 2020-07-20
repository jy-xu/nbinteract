import pandas as pd

import os
import PIL

from ipywidgets import Button, ToggleButtons, Layout, Output, ToggleButtonsStyle
from ipywidgets import Dropdown, IntSlider, HBox, interactive, IntText
from IPython.display import display
import asyncio

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback
        self._task = asyncio.ensure_future(self._job())

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def cancel(self):
        self._task.cancel()

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
        return debounced
    return decorator

def active_learning(root_path):
    image_dir = root_path + 'test_full'

    df_test_outputs = pd.read_csv(root_path + 'test_outputs.csv')
    # Create widgets
    image_list = [""] + [jpg.split(".")[0] for jpg in sorted(os.listdir(image_dir))]
    images = Dropdown(description="image_name", options=image_list, layout=Layout(width='220px'))
    df_error = df_test_outputs[(df_test_outputs.image_name == images.value)]
    error_list = df_error.error_id[df_error.error_id != -1]
    errors = Dropdown(description="error_id", options=error_list,
                      layout=Layout(width='150px', description_width='0px'))
    grid_id = IntText(description="grid_id", value=0, layout=Layout(width='150px'))
    slider_x = IntSlider(description="grid_x", min=0, max=4000, step=200, value=0,
                         layout=Layout(width='450px'), style={"description_width": '40px'})
    slider_y = IntSlider(description="grid_y", min=0, max=2000, step=200, value=0, layout=Layout(height='400px'),
                         orientation='vertical')
    togglebutton = ToggleButtons(description=" ", value=0, options={"artifact": 1, "no artifact": 0},
                                 style=ToggleButtonsStyle(button_width='90px', description_width='0px'))
    prevbutton = Button(description='Prev', layout=Layout(height='30px', width='80px'))
    nextbutton = Button(description='Next', layout=Layout(height='30px', width='80px'))

    def update_errors_options(*args):
        df_error = df_test_outputs[(df_test_outputs.image_name == images.value)]
        errors.options = df_error.error_id[df_error.error_id != -1]
        errors.value = errors.options[0]

    @debounce(0.3)
    def update_grid_id(*args):
        df_error = df_test_outputs[(df_test_outputs.image_name == images.value)]
        grid_id.value = df_error.grid_id[(df_error.grid_x == slider_x.value) & (df_error.grid_y == slider_y.value)].values[
            0]

    def update_slider(*args):
        df_error = df_test_outputs[(df_test_outputs.image_name == images.value)]
        slider_y.value, slider_x.value = tuple(df_error.loc[df_error.error_id == errors.value,
                                                            ['grid_y', 'grid_x']].values[0])

    @debounce(0.3)
    def update_errors_value(*args):
        df_error = df_test_outputs[(df_test_outputs.image_name == images.value)]
        df_error_id = df_error.error_id[(df_error.grid_x == slider_x.value) & (df_error.grid_y == slider_y.value)].values[0]
        errors.value = df_error_id if df_error_id != -1 else errors.value

    def next_error(*args):
        if errors.value != errors.options[-1]:
            errors.value += 1
        else:
            images.value = images.options[images.options.index(images.value) + 1]
            errors.value = errors.options[0]

    def prev_error(*args):
        if errors.value != errors.options[0]:
            errors.value -= 1
        else:
            images.value = images.options[images.options.index(images.value) - 1]
            errors.value = errors.options[-1]

    def get_label(*args):
        df_error = df_test_outputs[(df_test_outputs.image_name == images.value)]
        togglebutton.value = df_error.label_new[df_error.grid_id == grid_id.value].values[0]

    prevbutton.on_click(prev_error)
    nextbutton.on_click(next_error)

    images.observe(update_errors_options, 'value')  # change error_id dropdown list if image_name changes
    errors.observe(update_slider, 'value')  # change sliders if error_id changes
    slider_x.observe(update_grid_id, 'value')  # change grid_id if sliders changes
    slider_y.observe(update_grid_id, 'value')  # change grid_id if sliders changes
    grid_id.observe(get_label, 'value')  # change label if grid_id changes
    slider_x.observe(update_errors_value, 'value')  # change grid_id if sliders changes
    slider_y.observe(update_errors_value, 'value')  # change grid_id if sliders changes

    output1 = Output(layout={'width': '400px', 'height': '400px'})
    output2 = Output(layout={'width': '700px', 'height': '80px'})

    def get_errors(image_name, grid_id):
        display_columns = ['error_id', 'grid_id', 'image_name', 'grid_x', 'grid_y', 'label',
                           'preds', 'scores', 'confmat_labels', 'label_new']
        df_error = df_test_outputs[df_test_outputs.image_name == image_name]
        df_error_current = df_error.loc[df_error.grid_id == grid_id, display_columns].style.hide_index()
        with output2:
            output2.clear_output()
            display(df_error_current)

    def update_label(image_name, grid_id, artifact):
        df_test_outputs.loc[(df_test_outputs.image_name == image_name)
                            & (df_test_outputs.grid_id == grid_id), 'label_new'] = artifact
        display_columns = ['error_id', 'grid_id', 'image_name', 'grid_x', 'grid_y', 'label',
                           'preds', 'scores', 'confmat_labels', 'label_new']
        df_error = df_test_outputs[df_test_outputs.image_name == image_name]
        df_error_current = df_error.loc[df_error.grid_id == grid_id, display_columns].style.hide_index()
        with output2:
            output2.clear_output()
            display(df_error_current)

    # Show the images
    def show_images(image_name, grid_id):
        df_error = df_test_outputs[df_test_outputs.image_name == image_name]
        grid_y = df_error.grid_y[df_error.grid_id == grid_id].values[0]
        grid_x = df_error.grid_x[df_error.grid_id == grid_id].values[0]
        with output1:
            output1.clear_output()
            image_path = f"{image_dir}/{image_name}.jpg"
            offset = 0
            image = PIL.Image.open(image_path)
            image = image.crop((grid_x - offset, grid_y - offset, grid_x + 200 + offset, grid_y + 200 + offset))
            image = image.resize((400, 400))
            display(image)

    w = interactive(show_images, image_name=images, grid_id=grid_id)
    w2 = interactive(get_errors, image_name=images, grid_id=grid_id)
    w3 = interactive(update_label, image_name=images, grid_id=grid_id, artifact=togglebutton)

    box_layout = Layout(display='flex', flex_flow='row', justify_content='flex-start', align_items='center')
    display(HBox([images, errors, prevbutton, nextbutton], layout=box_layout))
    box_layout = Layout(display='flex', flex_flow='row', justify_content='flex-start', align_items='center')
    display(HBox([output2], layout=box_layout))
    box_layout = Layout(display='margin-left', flex_flow='row', justify_content='flex-start', align_items='flex-end')
    display(HBox([slider_x, togglebutton], layout=box_layout))
    box_layout = Layout(display='flex', flex_flow='row', justify_content='flex-start', align_items='flex-start')
    display(HBox([output1, slider_y], layout=box_layout))