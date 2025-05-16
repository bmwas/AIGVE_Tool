from vbench.utils import get_prompt_from_filename, init_submodules, save_json, load_json
from vbench import VBench
import importlib




class VBenchwithReturn(VBench):
    def __init__(self, device, full_info_dir):
        super(VBenchwithReturn, self).__init__(device=device, full_info_dir=full_info_dir, output_path='./output')

    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False,
                 mode='vbench_standard', **kwargs):

        print(f'videos_path: {videos_path}')
        print(f'prompt_list: {prompt_list}')
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode,
                                                       **kwargs)

        for dimension in dimension_list:
            print(f'Processing dimension: {dimension}')
            try:
                dimension_module = importlib.import_module(f'vbench.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        return results_dict