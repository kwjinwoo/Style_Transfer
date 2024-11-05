import torch
from PIL import Image
from torchvision.transforms import transforms

from style_transfer.image_processor import ImageProcessor


def test_image_processor(config):
    content_img_path = "./img/picasso.jpg"
    style_img_path = "./img/dancing.jpg"

    test_content = Image.open(content_img_path)
    test_style = Image.open(style_img_path)

    transformer = transforms.Compose(
        [transforms.Resize(512), transforms.ToTensor(), transforms.Normalize([2, 2, 2], [0.5, 0.5, 0.5])]
    )
    test_content_norm = transformer(test_content)
    test_style_norm = transformer(test_style)

    image_processor = ImageProcessor(content_img_path, style_img_path, config)

    content_img, style_img, gen_img = image_processor.get_train_images()

    assert content_img.size() == torch.Size([1, 3, 512, 512])
    assert style_img.size() == torch.Size([1, 3, 512, 512])
    assert gen_img.size() == torch.Size([1, 3, 512, 512])

    assert torch.allclose(test_content_norm, content_img, rtol=1e-3, atol=1e-5)
    assert torch.allclose(test_style_norm, style_img, rtol=1e-3, atol=1e-5)
    assert torch.allclose(test_content_norm, gen_img, rtol=1e-3, atol=1e-5)

    to_tensor = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    test_content = to_tensor(test_content)
    test_style = to_tensor(test_style)

    post_content_img = image_processor.post_processing(content_img)
    post_style_img = image_processor.post_processing(style_img)
    post_gen_img = image_processor.post_processing(gen_img)

    assert torch.allclose(post_content_img, test_content, rtol=1e-3, atol=1e-5)
    assert torch.allclose(post_style_img, test_style, rtol=1e-3, atol=1e-5)
    assert torch.allclose(post_gen_img, test_content, rtol=1e-3, atol=1e-5)
