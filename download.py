import urllib.request
import os
import logging
import zipfile
import sys
def download(url, dirpath):
    logger = logging.getLogger()
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    try:
        u = urllib.request.urlopen(url)
    except:
        logger.info("URL %s failed to open" % url)
        raise Exception
    try:
        f = open(filepath, 'wb')
    except:
        logger.info("Cannot write %s" % filepath)
        raise Exception
    try:
        filesize = int(u.info().get("Content-Length"))
    except:
        logger.info("URL %s failed to report length" % url)
        raise Exception
    logger.info("Downloading: %s Bytes: %s" % (filename, filesize))
    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
                  ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath
def unzip(filepath):
    logger = logging.getLogger()
    logger.info("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)
def download_sick(dirpath):
    logger = logging.getLogger()
    if os.path.exists(dirpath):
        logger.info('Found SICK dataset - skip,path:%s'%dirpath)
        return
    else:
        os.makedirs(dirpath)
    train_url = 'http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip'
    trial_url = 'http://alt.qcri.org/semeval2014/task1/data/uploads/sick_trial.zip'
    test_url = 'http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip'
    unzip(download(train_url, dirpath))
    unzip(download(trial_url, dirpath))
    unzip(download(test_url, dirpath))
def download_wordvecs(dirpath):
    logger = logging.getLogger()
    if os.path.exists(dirpath):
        logger.info('Found Glove vectors - skip,path:%s'%dirpath)
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))
