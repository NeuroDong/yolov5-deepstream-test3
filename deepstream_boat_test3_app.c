/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include "gstnvdsmeta.h"
//#include "gstnvstreammeta.h"
#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* By default, OSD process-mode is set to CPU_MODE. To change mode, set as:
 * 1: GPU mode (for Tesla only)
 * 2: HW mode (For Jetson only)
 */
#define OSD_PROCESS_MODE 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "RoadSign"
};

#define FPS_PRINT_INTERVAL 300
//static struct timeval start_time = { };

//static guint probe_counter = 0;

/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

typedef struct xy_xy
{
float x1;
float y1;
float x2;
float y2;
}xyxy;

xyxy xywh2xyxy(float x,float y,float w,float h)
{
  xyxy det_box_xyxy;
  det_box_xyxy.x1 = x - (w/2);
  det_box_xyxy.y1 = y - (h/2);
  det_box_xyxy.x2 = x + (w/2);
  det_box_xyxy.y2 = y + (h/2);
  return det_box_xyxy;
}

float IOU(float x1,float y1,float x2,float y2,float x3,float y3,float x4,float y4)
{
  float min(float a,float b)
  {
    if (a > b)
    {
      return b;
    }
    else
    {
      return a;
    }
  }
  float max(float a,float b)
  {
    if (a < b)
    {
      return b;
    }
    else
    {
      return a;
    }
  }
  float w,H,SA,SB,cross;
  w = min(x2,x4) - max(x1,x3);
  H = min(y2,y4) - max(y1,y3);
  if ((w <= 0) || (H <= 0)) return 0;
  SA = (x2-x1) * (y2-y1);
  SB = (x4-x3) * (y4-y3);
  cross = w * H;
  return cross/(SA + SB - cross);
}

static GstPadProbeReturn
tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    //定义用于记录前框的结构体数组
    typedef struct boat_Front_chuanti_frame 
    {
      float x1;
      float y1;
      float x2;
      float y2;
      int direction;
      int center_coord_length;
      float center_coord[2000];//没有赋初值的部分全为0，最开头的索引记录的是数组非零的长度
    }boat_Front_chuanti_frames[200],linshi_data;
    boat_Front_chuanti_frames boat_Front_chuanti_queding;
    //g_print("struct_x1:%f",boat_Front_chuanti_queding[0].x1);
    //初始化进船出船数
    int downward_counts;
    int upward_counts;
    int Front_frame_det_number;// 记录前一帧中检测框的数量
    int det_boxe_number;

    //获取从管道中获取推理结果
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    //初始化要使用的数据结构
    NvDsObjectMeta *obj_meta = NULL; //目标检测元数据类型变量
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL; 


    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;l_frame = l_frame->next) //从批量中获取某一帧图
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        //获取图片的宽和高，并且设置计数线的位置
        guint x_coord,y_coord,jishu_line;
        x_coord = 1920;
        y_coord = 1080;
        //x_coord = frame_meta -> source_frame_width;
        //y_coord = frame_meta -> source_frame_height;
        //g_print("x_coord:%d",x_coord);
        //g_print("y_coord:%d",y_coord);
        //g_print("frame_width:%d frame_height:%d \n",x_coord,y_coord);
        jishu_line = x_coord/6;

        boat_Front_chuanti_frames Front_det_boxes={0};
        det_boxe_number=0;
        if (frame_number==0) //判断是不是第一帧
        {
          //g_print("come in \n");
          //获取图片帧里面的检测框并进行处理
          for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;l_obj = l_obj->next) 
          {
              //g_print("struct_x1:%f \n",Front_det_boxes[199].x1);
              //获取检测框的坐标信息
              obj_meta = (NvDsObjectMeta *) (l_obj->data);
              if (obj_meta->class_id == 2) //2为车
              {
                g_print("obj_meta:%d \n",obj_meta->class_id);
                NvDsComp_BboxInfo boxInfo;
                boxInfo = obj_meta->detector_bbox_info;
                NvBbox_Coords box_Coord;
                box_Coord = boxInfo.org_bbox_coords;
                float left,top,width,height;
                left = box_Coord.left;top = box_Coord.top;width = box_Coord.width;height = box_Coord.height;
                //g_print("xywh:(%f,%f,%f,%f) \n",left,top,width,height);
                xyxy det_boxe = xywh2xyxy(left,top,width,height);
                //g_print("det_boxe:(%f,%f,%f,%f) \n",det_boxe.x1,det_boxe.y1,det_boxe.x2,det_boxe.y2);
                //获取框的中心点坐标
                float x,y;
                x=(det_boxe.x1+det_boxe.x2)/2;
                y=(det_boxe.y1+det_boxe.y2)/2;

                linshi_data ss={0};

                if (y < jishu_line)
                {
                  //g_print("forward right \n");
                  ss.x1 = det_boxe.x1;
                  ss.y1 = det_boxe.y1;
                  ss.x2 = det_boxe.x2;
                  ss.y2 = det_boxe.y2;
                  ss.direction = 0; //代表向下
                  ss.center_coord[0] = x;
                  ss.center_coord[1] = y;
                  ss.center_coord_length = 2;
                }
                else 
                {
                  //g_print("forward left \n");
                  ss.x1 = det_boxe.x1;
                  ss.y1 = det_boxe.y1;
                  ss.x2 = det_boxe.x2;
                  ss.y2 = det_boxe.y2;
                  ss.direction = 1; //代表向上
                  ss.center_coord[0] = x;
                  ss.center_coord[1] = y;
                  ss.center_coord_length = 2;
                }
              Front_det_boxes[det_boxe_number] = ss;
              det_boxe_number = det_boxe_number + 1;
              }   
          }
        }
        else
        {
          //获取图片帧里面的检测框并进行处理
          for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;l_obj = l_obj->next) 
          {
            //获取检测框的坐标信息
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == 2) //2为车
            {
              g_print("obj_meta:%d \n",obj_meta->class_id);
              NvDsComp_BboxInfo boxInfo;
              boxInfo = obj_meta->detector_bbox_info;
              NvBbox_Coords box_Coord;
              box_Coord = boxInfo.org_bbox_coords;
              float left,top,width,height;
              left = box_Coord.left;top = box_Coord.top;width = box_Coord.width;height = box_Coord.height;
              //g_print("xywh:(%f,%f,%f,%f) \n",left,top,width,height);
              xyxy det_boxe = xywh2xyxy(left,top,width,height);
              //g_print("det_boxe:(%f,%f,%f,%f) \n",det_boxe.x1,det_boxe.y1,det_boxe.x2,det_boxe.y2);

              int hh=0; //用于判断是不是新蹦出来的目标
              for (int i =0;i < Front_frame_det_number;i++)
              {
                linshi_data Front_det_boxe_xy=boat_Front_chuanti_queding[i];
                //g_print("IOU:%f \n",IOU(det_boxe.x1,det_boxe.y1,det_boxe.x2,det_boxe.y2,Front_det_boxe_xy.x1,Front_det_boxe_xy.y1,Front_det_boxe_xy.x2,Front_det_boxe_xy.y2));
                //计算IOU
                if (IOU(det_boxe.x1,det_boxe.y1,det_boxe.x2,det_boxe.y2,Front_det_boxe_xy.x1,Front_det_boxe_xy.y1,Front_det_boxe_xy.x2,Front_det_boxe_xy.y2) > 0.5)
                {
                  //更新检查框的位置 
                  Front_det_boxe_xy.x1=det_boxe.x1;
                  Front_det_boxe_xy.y1=det_boxe.y1;
                  Front_det_boxe_xy.x2=det_boxe.x2;
                  Front_det_boxe_xy.y2=det_boxe.y2;
                  hh = 1;
                  //加入中心的的坐标，且防止局部往回走 
                  //获取框的中心点坐标 
                  float x,y;
                  x=(det_boxe.x1+det_boxe.x2)/2;
                  y=(det_boxe.y1+det_boxe.y2)/2;
                  if (Front_det_boxe_xy.direction ==0) //向下走
                  {
                    if (Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length-1] > y)
                    {
                      //g_print("center_coord_length1:%d \n",Front_det_boxe_xy.center_coord_length);
                      Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length] = Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length-2];
                      Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length+1] = Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length-1];
                      Front_det_boxe_xy.center_coord_length = Front_det_boxe_xy.center_coord_length + 2;
                    }
                    else
                    {
                      //g_print("center_coord_length2:%d \n",Front_det_boxe_xy.center_coord_length);
                      Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length] = x;
                      Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length+1] = y;
                      Front_det_boxe_xy.center_coord_length = Front_det_boxe_xy.center_coord_length + 2;
                    }
                  }
                  else //向上走
                  {
                    if (Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length-1] < y)
                    {
                      //g_print("center_coord_length3:%d \n",Front_det_boxe_xy.center_coord_length);
                      Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length] = Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length-2];
                      Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length+1] = Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length-1];
                      Front_det_boxe_xy.center_coord_length = Front_det_boxe_xy.center_coord_length + 2;
                    }
                    else
                    {
                      //g_print("center_coord_length4:%d \n",Front_det_boxe_xy.center_coord_length);
                      Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length] = x;
                      Front_det_boxe_xy.center_coord[Front_det_boxe_xy.center_coord_length+1] = y;
                      Front_det_boxe_xy.center_coord_length = Front_det_boxe_xy.center_coord_length + 2;
                    }
                  }
                  //g_print("here \n");
                  g_print("det_boxe_number:%d \n",det_boxe_number);
                  Front_det_boxes[det_boxe_number] = Front_det_boxe_xy;
                  det_boxe_number = det_boxe_number + 1;
                 }

               }
              if (hh == 0)
              {
                //获取框的中心点坐标
                float x,y;
                x=(det_boxe.x1+det_boxe.x2)/2;
                y=(det_boxe.y1+det_boxe.y2)/2;
                linshi_data ss={0};

                if (y < jishu_line)
                {
                  //g_print("forward right \n");
                  ss.x1 = det_boxe.x1;
                  ss.y1 = det_boxe.y1;
                  ss.x2 = det_boxe.x2;
                  ss.y2 = det_boxe.y2;
                  ss.direction = 0; //代表向下
                  ss.center_coord[0] = x;
                  ss.center_coord[1] = y;
                  ss.center_coord_length = 2;
                }
                else
                {
                  //g_print("forward left \n");
                  ss.x1 = det_boxe.x1;
                  ss.y1 = det_boxe.y1;
                  ss.x2 = det_boxe.x2;
                  ss.y2 = det_boxe.y2;
                  ss.direction = 1; //代表向上
                  ss.center_coord[0] = x;
                  ss.center_coord[1] = y;
                  ss.center_coord_length = 2;
                }
                Front_det_boxes[det_boxe_number] = ss;
                det_boxe_number = det_boxe_number + 1;
              }
            }
            
            //g_print("hh:%d \n",hh);
          }
        }
        memcpy(boat_Front_chuanti_queding,Front_det_boxes,sizeof(Front_det_boxes));
        Front_frame_det_number=det_boxe_number;
        g_print("Front_frame_det_number:%d \n",Front_frame_det_number);

        //船舶计数
        for (int i =0;i < Front_frame_det_number;i++)
        {
          linshi_data boat_counts_frame = boat_Front_chuanti_queding[i];
          g_print("center_coord_length:%d \n",boat_counts_frame.center_coord_length);
          if (boat_counts_frame.center_coord_length >= 4)
          {
            float center_x1,center_y1,center_x2,center_y2;
            center_x1 = boat_counts_frame.center_coord[boat_counts_frame.center_coord_length-2];
            center_y1 = boat_counts_frame.center_coord[boat_counts_frame.center_coord_length-1];
            center_x2 = boat_counts_frame.center_coord[boat_counts_frame.center_coord_length-4];
            center_y2 = boat_counts_frame.center_coord[boat_counts_frame.center_coord_length-3];
            //进入的船舶数
            if ((center_y1 > jishu_line) && (center_y2 <= jishu_line))
            {
              downward_counts = downward_counts + 1;
              //获取时间，存入文件

            }
            //出去的船舶数
            if ((center_y1 < jishu_line) && (center_y2 >= jishu_line))
            {
              upward_counts = upward_counts + 1;
            }
          }
        }

        //画左上角的统计信息
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        int offset = 0;
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "downward_counts= %d \n", downward_counts);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "upward_counts = %d ", upward_counts);
        txt_params->x_offset = 30;
        txt_params->y_offset = 30;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 30;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        //画线
        NvOSD_LineParams *line_params = &display_meta -> line_params[0];
        display_meta->num_lines = 1;
        line_params -> x1 = 0;
        line_params -> y1 = jishu_line;
        line_params -> x2 = x_coord;
        line_params -> y2 = jishu_line;
        line_params -> line_width = 5;
        line_params -> line_color.red = 0.0;
        line_params -> line_color.green = 1.0;
        line_params -> line_color.blue = 0.0;
        line_params -> line_color.alpha = 1.0;

        
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    g_print ("Frame Number = %d \n",frame_number);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
#ifndef PLATFORM_TEGRA
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
#endif
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
      *queue1, *queue2, *queue3, *queue4, *queue5, *nvvidconv = NULL,
      *nvosd = NULL, *tiler = NULL;
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_src_pad = NULL;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
    return -1;
  }
  num_sources = argc - 1;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest3-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };
    GstElement *source_bin = create_source_bin (i, argv[i + 1]);

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Use nvinfer to infer on batched frame. */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!pgie || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

#ifdef PLATFORM_TEGRA
  if(!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "config_infer_primary_yoloV5.txt", NULL);

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
      "display-text", OSD_DISPLAY_TEXT, NULL);

  g_object_set (G_OBJECT (sink), "qos", 0, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
  gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, tiler, queue3,
      nvvidconv, queue4, nvosd, queue5, transform, sink, NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, queue1, pgie, queue2, tiler, queue3,
        nvvidconv, queue4, nvosd, queue5, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#else
gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, tiler, queue3,
    nvvidconv, queue4, nvosd, queue5, sink, NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, queue1, pgie, queue2, tiler, queue3,
        nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#endif

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  tiler_src_pad = gst_element_get_static_pad (pgie, "src");
  if (!tiler_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (tiler_src_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[i + 1]);
  }
  g_print ("\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
