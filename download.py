from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

pipe = pipeline(
    # task=Tasks.text_to_image_synthesis,
    model='black-forest-labs/FLUX.1-dev',
    cachre_dir='/root/user-data/shrcode/omini_sod/flux2'
)


#modelscope download --model stabilityai/stable-diffusion-3.5-medium --local_dir /root/user-data/shrcode/omini_sod/sd3.5