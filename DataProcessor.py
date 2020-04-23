class DataProcessor:
    def __init__(self, data_route, bg_out_route = "./0/", fg_out_route = "./1/"):
        self.data_route = data_route
        self.bg_out_route = bg_out_route
        self.fg_out_route = fg_out_route
        self.count = 0
        
    def rectCoinside(self, rect_big, rect_small):
        center_big = [ (rect_big[0] + rect_big[2]) / 2, (rect_big[1] + rect_big[3]) / 2 ]
        center_small = [ (rect_small[0] + rect_small[2]) / 2, (rect_small[1] + rect_small[3]) / 2 ]
        width_between_centers = abs(center_big[0] - center_small[0])
        height_between_centers = abs(center_big[1] - center_small[1])
        half_width_big = abs((rect_big[0] - rect_big[2]) / 2)
        half_height_big = abs((rect_big[1] - rect_big[3]) / 2)
        half_width_small = abs((rect_small[0] - rect_small[2]) / 2)
        half_height_small = abs((rect_small[1] - rect_small[3]) / 2)
        if half_width_big + half_width_small > width_between_centers and half_height_big + half_height_small > height_between_centers:
            area_small = half_width_small * half_height_small * 4
            coinside_width = min(rect_big[2], rect_small[2]) - max(rect_big[0], rect_small[0])
            coinside_height = min(rect_big[3], rect_small[3]) - max(rect_big[1], rect_small[1])
            coinside_area = coinside_width * coinside_height
            coincide_rate = coinside_area / area_small
            return coincide_rate
        else:
            return 0
    
    def judgeClass(self, rect_big, rect_small):
        result = self.rectCoinside(rect_big, rect_small)
        coincide_rate = self.rectCoinside(rect_big, rect_small)
    #     print(str(coincide_rate))
        if coincide_rate < 0.5:
            #background
            return 0
        elif coincide_rate >= 0.5:
            #foreground
            return 1
    
    def slidWindows(self, img_name, x_divide, y_divide, x_strip, y_strip, x1, y1, x2, y2 ):
        count = 0
        img = Image.open(img_name)
        width, height = img.size
        x_current, y_current = [0, 0]
        while y_current < height:
            while x_current < width:
                #crop the image

                x_crop = x_current + x_divide
                y_crop = y_current + y_divide
                crop_box = [x_current, y_current, x_crop, y_crop]
                crop = img.crop(box = crop_box)

                img_class = self.judgeClass([x1, y1, x2, y2], [x_current, y_current, x_crop, y_crop])
    #             print(str(img_class))
                name = str(self.count) + ".jpg"
                filename = img_name.split("/")[-1].split(".")[0]
                name = filename + str(count) + ".jpg"
            
                if img_class == 0:
                    crop.save(self.bg_out_route + name)
                else:
                    crop.save(self.fg_out_route + name)
                    
                count += 1
                x_current += x_strip
            x_current = 0
            y_current += y_strip
            
    def process(self, x_divide, y_divide, x_strip, y_strip):
        data = os.walk(self.data_route)
        filenames = []
        for root, dirs, files in data:
            for name in files:
                filenames.append(name)
                
        print(filenames)
        for name in filenames:
            prename, sufname = name.split(".")
            if(sufname == "jpg"):
                label_file = self.data_route + prename + ".json"
                label = []
                with open(label_file, "r") as f:
                    tmp = json.load(f)
                    label = tmp['shapes'][0]['points']
                label = label[0] + label[1]
                #print(label)
                img_name = self.data_route + name
                self.slidWindows(img_name, x_divide, y_divide, x_strip, y_strip, *label)
            
crop = DataProcessor("./crop/", "dataset2/0/", "dataset2/1/")
crop.process(128, 128, 100, 100)