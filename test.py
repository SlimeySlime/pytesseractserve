

img_to_show = []

img_to_show.append({
    'img': 'img_exampled',
    'title': 'title1'
})
img_to_show.append({
    'img': 'img_example2',
    'title': 'title2'
})


def show_images(img_list):
    for i, item in enumerate(img_list):
        print(f'item is {item}')
        print(item['title'])
        print(item['img'])

    return

show_images(img_to_show)
