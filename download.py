from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipe = pipeline(
    # task=Tasks.text_to_image_synthesis,
    model='black-forest-labs/FLUX.1-dev',
    cachre_dir='/home/shenhaoran/project/project/OminiControl-main/flux2'
)