import json
import tempfile

import numpy as np
import copy
import time
import torch
import torch._six
import datetime

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from collections import defaultdict

from references import utils


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].params.maxDets = [100]
            self.coco_eval[iou_type].params.areaRng = [[0 ** 2, 1e5 ** 2]]
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iou_types[0]) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.params.imgIds = sorted(coco_gt.getImgIds())
        self.params.catIds = sorted(coco_gt.getCatIds())

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self, p=None):
        #for coco_eval in self.coco_eval.values():
        #    coco_eval.accumulate()

        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.eval_imgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params        
        self._paramsEval = copy.deepcopy(self.params)
        iou_type = 'bbox'
        evalImgs = self.coco_eval[iou_type].evalImgs
        p = copy.deepcopy(self.coco_eval[iou_type].params)
        self._paramsEval = copy.deepcopy(self.coco_eval[iou_type]._paramsEval)
        #print(evalImgs.shape)
        #print('Len CatIds ====== ', len(self.params.catIds))
        p.catIds = self.params.catIds# if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) #if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        print(p.iouThrs, p.recThrs, p.catIds, p.areaRng, p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = self.params.catIds # catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        #print(setK)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list): #Classes
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list): # Area ranges
                Na = a0*I0
                for m, maxDet in enumerate(m_list): # Max Detections
                    E = [evalImgs[Nk + Na + i] for i in i_list] # Image Ids
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval[iou_type] = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))


    def summarize(self, output_details=None):
        #for iou_type, coco_eval in self.coco_eval.items():
        #    print("IoU metric: {}".format(iou_type))
        #    coco_eval.summarize()

        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100, iou_type='bbox', output_details=None):
            p = copy.deepcopy(self.coco_eval[iou_type].params)
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval[iou_type]['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval[iou_type]['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            #print('s issssss', s)
            num_classes = len(p.catIds)
            if len(s[s>-1])==0:
                mean_s = -np.ones(num_classes + 1)
            else:
                #mean_s = np.mean(s[s>-1])
                #cacluate AP(average precision) for each category
                mean_s = -np.ones(num_classes + 1)
                avg_ap = 0.0
                if ap == 1:
                    for i in range(0, num_classes):
                        print('category : {0} : {1}'.format(i,np.mean(s[:,:,i,:])))
                        mean_s[i] = np.mean(s[:,:,i,:])
                        #avg_ap +=np.mean(s[:,:,i,:])
                    #print('(all categories) mAP : {}'.format(avg_ap / num_classes))
                else:
                    for i in range(0, num_classes):
                        print('category : {0} : {1}'.format(i,np.mean(s[:,i,:])))
                        mean_s[i] = np.mean(s[:,i,:])
                        #avg_ap +=np.mean(s[:,i,:])
                    #print('(all categories) mAR : {}'.format(avg_ap / num_classes))
                mean_s[-1] = np.mean(s[s>-1])

            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s[-1]))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,len(self.params.catIds) + 1))
            stats[0,:] = _summarize(1)
            stats[1,:] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[0], output_details=output_details)
            stats[2,:] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[0], output_details=output_details)
            stats[3,:] = _summarize(1, iouThr=.85, maxDets=self.params.maxDets[0], output_details=output_details)
            stats[4,:] = _summarize(1, iouThr=.90, maxDets=self.params.maxDets[0], output_details=output_details)
            stats[5,:] = _summarize(1, iouThr=.95, maxDets=self.params.maxDets[0], output_details=output_details)
            #stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2], output_details=output_details)
            #stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2], output_details=output_details)
            #stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2], output_details=output_details)

            stats[6,:] = _summarize(0)
            stats[7,:] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[0], output_details=output_details)
            stats[8,:] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[0], output_details=output_details)
            stats[9,:] = _summarize(0, iouThr=.85, maxDets=self.params.maxDets[0], output_details=output_details)
            stats[10,:] = _summarize(0, iouThr=.90, maxDets=self.params.maxDets[0], output_details=output_details)
            stats[11,:] = _summarize(0, iouThr=.95, maxDets=self.params.maxDets[0], output_details=output_details)
            #stats[6] = _summarize(0, maxDets=self.params.maxDets[0], output_details=output_details)
            #stats[7] = _summarize(0, maxDets=self.params.maxDets[0], output_details=output_details)
            #stats[8] = _summarize(0, maxDets=self.params.maxDets[0], output_details=output_details)
            #stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2], output_details=output_details)
            #stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2], output_details=output_details)
            #stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2], output_details=output_details)
            return stats


        if not self.eval:
            raise Exception('Please run accumulate() first')
        
        iouType = self.params.iouType
        
        summarize = _summarizeDets
        self.stats = summarize()

    def __str__(self):
        self.summarize()


    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    #print(eval_imgs.shape)
    eval_imgs = list(eval_imgs.flatten())
    #print(len(eval_imgs))
    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions

def createIndex(self):
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = mask_util


def loadRes(self, resFile, bbox = []):
    """
    Load result file and return a result api object.
    Args:
        self (obj): coco object with ground truth annotations
        resFile (str): file name of result file
    Returns:
    res (obj): result api object
    """
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for id, ann in enumerate(anns):
            if bbox:
                x1, x2, y1, y2 = [bbox[0], bbox[2], bbox[1], bbox[3]]
                ann['area'] = (x2 - x1)*(y2 - y1)
            else:
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                ann['area'] = bb[2] * bb[3]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset['annotations'] = anns
    createIndex(res)
    return res


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    p.iouType = 'bbox'
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    self.ious = {
        (imgId, catId): self.computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    #print('ious', self.ious)
    #print('evalimgs', evalImgs)
    #print('evalimgs 0', evalImgs.shape)
    #self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################


class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        #self.maxDets = [1, 10, 100]
        self.maxDets = [100]
        #self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRng = [[0 ** 2, 1e5 ** 2]]
        #self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.areaRngLbl = ['all']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None