// [20151223_Curtis] this header file is modified from lbp.h of dlib to support multi-scale uniform lbp

#include <dlib/image_processing/full_object_detection.h>
#include <dlib/image_transforms/lbp.h>

namespace dlib
{
   template <
      typename image_type,
      typename image_type2
   >
   void make_uniform_lbp_image(
   const image_type& img_,
   image_type2& lbp_,
   const int scale
   )
   {
      const static unsigned char uniform_lbps[] = {
         0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58,
         58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58,
         58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58,
         20, 21, 22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
         58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58,
         58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58,
         58, 58, 26, 58, 27, 28, 29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58,
         58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 58, 58,
         58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
         58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58,
         58, 58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
         58, 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48,
         58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57
      };

      COMPILE_TIME_ASSERT(sizeof(uniform_lbps) == 256);

      const_image_view<image_type> img(img_);
      image_view<image_type2> lbp(lbp_);

      lbp.set_size(img.nr(), img.nc());

      // set all the border pixels to the "non-uniform LBP value".
      assign_border_pixels(lbp, scale, scale, 58);

      typedef typename image_traits<image_type>::pixel_type pixel_type;
      typedef typename pixel_traits<pixel_type>::basic_pixel_type basic_pixel_type;

      for (long r = scale; r + scale < img.nr(); ++r)
      {
         for (long c = scale; c + scale < img.nc(); ++c)
         {
            const basic_pixel_type pix = get_pixel_intensity(img[r][c]);
            unsigned char b1 = 0;
            unsigned char b2 = 0;
            unsigned char b3 = 0;
            unsigned char b4 = 0;
            unsigned char b5 = 0;
            unsigned char b6 = 0;
            unsigned char b7 = 0;
            unsigned char b8 = 0;

            unsigned char x = 0;
            if (get_pixel_intensity(img[r - scale][c - scale]) > pix) b1 = 0x80;
            if (get_pixel_intensity(img[r - scale][c]) > pix) b2 = 0x40;
            if (get_pixel_intensity(img[r - scale][c + scale]) > pix) b3 = 0x20;
            x |= b1;
            if (get_pixel_intensity(img[r][c - scale]) > pix) b4 = 0x10;
            x |= b2;
            if (get_pixel_intensity(img[r][c + scale]) > pix) b5 = 0x08;
            x |= b3;
            if (get_pixel_intensity(img[r + scale][c - scale]) > pix) b6 = 0x04;
            x |= b4;
            if (get_pixel_intensity(img[r + scale][c]) > pix) b7 = 0x02;
            x |= b5;
            if (get_pixel_intensity(img[r + scale][c + scale]) > pix) b8 = 0x01;

            x |= b6;
            x |= b7;
            x |= b8;

            lbp[r][c] = uniform_lbps[x];
         }
      }
   }

   // -------------------------------------------
   
   template <
      typename image_type,
      typename T
   >
   void extract_uniform_lbp_descriptors(
   const image_type& img,
   std::vector<T>& feats,
   const int scale,
   const unsigned int cell_size = 10
   )
   {
      // make sure requires clause is not broken
      DLIB_ASSERT(cell_size >= 1,
         "\t void extract_uniform_lbp_descriptors()"
         << "\n\t Invalid inputs were given to this function."
         << "\n\t cell_size:      " << cell_size
         );

      feats.clear();
      array2d<unsigned char> lbp;
      make_uniform_lbp_image(img, lbp, scale);
      for (long r = 0; r < lbp.nr(); r += cell_size)
      {
         for (long c = 0; c < lbp.nc(); c += cell_size)
         {
            const rectangle cell = rectangle(c, r, c + cell_size - 1, r + cell_size - 1).intersect(get_rect(lbp));
            // make the actual histogram for this cell
            unsigned int hist[59] = { 0 };
            for (long r = cell.top(); r <= cell.bottom(); ++r)
            {
               for (long c = cell.left(); c <= cell.right(); ++c)
               {
                  hist[lbp[r][c]]++;
               }
            }

            // copy histogram into the output.
            feats.insert(feats.end(), hist, hist + 59);
         }
      }

      //for (unsigned long i = 0; i < feats.size(); ++i)
      //{
      //   feats[i] = std::sqrt(feats[i]);
      //}
   }
}

