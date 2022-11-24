import os
from osgeo import gdal, ogr, gdalconst, osr
import shapefile
import numpy as np
from utils import mkdir


def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if (max_out <= 255):
            gray = np.uint8(gray)
        elif (max_out <= 65535):
            gray = np.uint16(gray)
        return gray

    #  如果是多波段
    if (len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)
    return image_stretch


def DivisionByProvince(inputfile, inputshape, outpath, name, n):
    # name = 内蒙.tif

    # or as an alternative if the input is already a gdal raster object you can use that gdal object
    input_raster = gdal.Open(inputfile)

    output_raster = outpath + '\\' + name  # your output raster file
    Cutlinewhere = 'FID' + ' = ' + str(n)
    gdal.Warp(output_raster,
              input_raster,
              format='GTiff',
              cutlineDSName=inputshape,  # or any other file format
              cutlineWhere=Cutlinewhere,
              # optionally you can filter your cutline (shapefile)
              )


def GetProvinceName(shapepath):
    """
    get province name
    :param shapepath: National provincial boundaries shapefile
    :return:Province Code | [Record #0: [110000, '北京市', '直辖市'], Record #1: [120000, '天津市', '直辖市'],.....
    """
    file = shapefile.Reader(shapepath)
    shapes = file.shapes()
    records = file.records()
    pro_points = []
    for i in range(len(shapes)):
        points = shapes[i].points

        lon = []
        lat = []
        # 将每个tuple的lon和lat组合起来
        [lon.append(points[i][0]) for i in range(len(points))]
        [lat.append(points[i][1]) for i in range(len(points))]

        lon = np.array(lon).reshape(-1, 1)
        lat = np.array(lat).reshape(-1, 1)
        loc = np.concatenate((lon, lat), axis=1)
        pro_points.append(loc)

    return records, pro_points


class IMAGE:

    # 读图像文件
    def read_img(self, filename):

        dataset = gdal.Open(filename)  # 打开文件

        # im_width = dataset.RasterXSize  # 栅格矩阵的列数
        # im_height = dataset.RasterYSize  # 栅格矩阵的行数
        # # im_bands = dataset.RasterCount  # 波段数
        # im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        # im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
        # dt = dataset.GetRasterBand(1)
        im_data = dataset.ReadAsArray()

        del dataset  # 关闭对象dataset，释放内存
        # return im_width, im_height, im_proj, im_geotrans, im_data,im_bands
        # return im_proj, im_geotrans, im_data, im_width, im_height, im_bands
        return im_data

    # 遥感影像的存储
    # 写GeoTiff文件
    def write_img(self, filename, im_data, im_proj=None, im_geotrans=None):
        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            # 注意数据的存储波段顺序：im_bands, im_height, im_width
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape  # 没看懂

        # 创建文件时 driver = gdal.GetDriverByName("GTiff")，数据类型必须要指定，因为要计算需要多大内存空间。
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        if im_geotrans is not None and im_proj is not None:
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset


if __name__ == "__main__":
    run = IMAGE()
    # data1 = run.read_img(r'E:\cloudremove\result\PSTCR/290-3.tif')  # 读数据
    # data2 = run.read_img(r'E:\cloudremove\result\PSTCR/215-3.tif')  # 读数据
    # data3 = run.read_img(r'E:\cloudremove\result\PSTCR/140-3.tif')  # 读数据
    # data = np.array((data1, data2, data3), dtype=data1.dtype)  # 按序将3个波段像元值放入
    # out = r'E:\cloudremove\result\PSTCR\140-3-color.tif'
    # run.write_img(out, data)
    outpath = r'D:\去云对比方法\WLR\wh\paper\20211211'
    intpath = r'D:\去云对比方法\WLR\kpl\1211result'
    pathlist = ['result']
    for pathname in pathlist:
        listdir = os.listdir(intpath + '\\' + pathname)
        if pathname == 'cloud':
            listdir.sort(key=lambda x: int(x[6:-4]))
        if pathname == 'gt':
            listdir.sort(key=lambda x: int(x[3:-4]))
        if pathname == 'result':
            listdir.sort(key=lambda x: int(x[7:-4]))
        outpath_ = outpath + '\\' + pathname
        if not os.path.exists(outpath_):
            mkdir(outpath_)
        length = int(len(listdir) / 4)
        # length = 75
        for idx, name in enumerate(listdir):
            # if length == idx:
            #     break
            if idx == 3:
                break
            n = name.split('_')
            # 2, 1 ,0 真彩色(432波段)， 3,2,1 红外彩色（5，4，3)波段
            data1 = run.read_img(intpath + '/' + pathname + '/' + pathname + '_' +
                                 str(int(n[1][:-4]) + length * 3) + name[-4:])  # nir
            data2 = run.read_img(intpath + '/' + pathname + '/' + pathname + '_' +
                                 str(int(n[1][:-4]) + length * 2) + name[-4:])  # Red
            data3 = run.read_img(intpath + '/' + pathname + '/' + pathname + '_' +
                                 str(int(n[1][:-4]) + length * 1) + name[-4:])  # Green

            out = outpath_ + '/' + name
            data = np.array((data1, data2, data3), dtype=data1.dtype)  # 按序将3个波段像元值放入
            data = truncated_linear_stretch(data, truncated_value=0.5, max_out=255, min_out=0)
            # 第三步
            run.write_img(out, data)
