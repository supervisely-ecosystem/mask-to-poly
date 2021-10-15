import globals as g
import supervisely_lib as sly
from supervisely_lib.geometry.polygon import Polygon
from bitmap_to_poly import get_polygon
import numpy as np
import cv2
from supervisely_lib.geometry.point_location import row_col_list_to_points


@g.my_app.callback("mask_to_poly")
@sly.timeit
def mask_to_poly(api: sly.Api, task_id, context, state, app_logger):

    project = api.project.get_info_by_id(g.PROJECT_ID)
    meta_json = api.project.get_meta(g.PROJECT_ID)
    meta = sly.ProjectMeta.from_json(meta_json)

    poly_obj_classes = []
    for obj_class in meta.obj_classes:
        poly_obj_class = obj_class.clone(geometry_type=Polygon)
        poly_obj_classes.append(poly_obj_class)

    poly_meta = meta.clone(obj_classes=sly.ObjClassCollection(poly_obj_classes))
    poly_meta_json = poly_meta.to_json()

    poly_project_name = project.name + g.poly_project_suffix
    poly_project_info = api.project.create(g.WORKSPACE_ID, poly_project_name, change_name_if_conflict=True)

    api.project.update_meta(poly_project_info.id, poly_meta_json)

    datasets = [ds for ds in api.dataset.get_list(g.PROJECT_ID)]

    for dataset in datasets:

        poly_ds_info = api.dataset.create(poly_project_info.id, dataset.name, change_name_if_conflict=True)

        images_infos = api.image.get_list(dataset.id)
        images_names = [im.name for im in images_infos]
        images_ids = [im.id for im in images_infos]

        ann_infos = api.annotation.download_batch(dataset.id, images_ids)
        anns = [sly.Annotation.from_json(x.annotation, meta) for x in ann_infos]

        poly_anns = []

        for ann in anns:
            poly_labels = []
            for label in ann.labels:
                curr_mask = np.zeros((ann.img_size[0], ann.img_size[1]), dtype=np.uint8)
                label.geometry.draw(curr_mask, 1)
                polygons = get_polygon(curr_mask)
                for poly in polygons:
                    curr_points = []
                    for p in poly:
                        curr_point = p.tolist()[::-1]
                        curr_points.append(curr_point)

                    curr_points = row_col_list_to_points(curr_points)
                    sl_poly = sly.Polygon(curr_points, interior=[])

                    poly_label = label.clone(geometry=sl_poly, obj_class=poly_meta.get_obj_class(label.obj_class.name))
                    poly_labels.append(poly_label)
            poly_ann = ann.clone(labels=poly_labels)
            poly_anns.append(poly_ann)

        new_img_infos = api.image.upload_ids(poly_ds_info.id, images_names, images_ids)
        img_ids = [img_info.id for img_info in new_img_infos]
        api.annotation.upload_anns(img_ids, poly_anns)




    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.TEAM_ID,
        "WORKSPACE_ID": g.WORKSPACE_ID,
        "PROJECT_ID": g.PROJECT_ID
    })
    g.my_app.run(initial_events=[{"command": "mask_to_poly"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)
