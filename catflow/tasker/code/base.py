import os
from ruamel.yaml import YAML
from abc import ABCMeta, abstractmethod


class BaseRender(metaclass=ABCMeta):
    @abstractmethod
    def from_file(self, path):
        pass

    @abstractmethod
    def make_file(self, path):
        pass


class TemplateRender(BaseRender):

    def __init__(self, render_dict):
        self.render_dict = render_dict
        self.template = None

    def from_file(self, template_file):
        from jinja2 import FileSystemLoader, Environment
        loader = FileSystemLoader(os.path.abspath(template_file))
        env = Environment(loader=loader)
        self.template = env.get_template('demo.html')

    def load_render_dict(self, render_config):
        yaml = YAML(typ='safe')
        with open(os.path.abspath(render_config)) as f:
            self.render_dict = yaml.load(f)

    def render_template(self):
        pass

    def make_file(self):
        pass



