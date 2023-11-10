#datadownload_iith.py
import datasets
from pathlib import Path
import numpy as np
from datasets.tasks import AutomaticSpeechRecognition
import os
import logging
import re

logging.basicConfig(level=logging.DEBUG, filename='iith.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

_DESCRIPTION = "IIITH Voice dataset"  # Assign a value to _DESCRIPTION

class IIITHDataset(datasets.GeneratorBasedBuilder):

  def _info(self):
      return datasets.DatasetInfo(
          description=_DESCRIPTION,
          features = datasets.Features(
                        {
                            "path": datasets.Value("string"),
                            "audio": datasets.Audio(sampling_rate=48_000),
                            "sentence": datasets.Value("string"),
                        })
      )
  
  def _split_generators(self, dl_manager):
        logging.info('XXXX Inside  _split_generators')

        archive_path = dl_manager.download(
            "https://drive.google.com/uc?export=download&id=1oRF33jUC1BkaKbMzndVSpX6GqM3dlPDY"
            #https://drive.google.com/file/d/1oRF33jUC1BkaKbMzndVSpX6GqM3dlPDY/view?usp=drive_link
        )
        logging.info('XXXX Inside  _split_generators')
        # First we locate the data using the path within the archive:
        path_to_data = "/".join(["iiit_hin","iiit_hin"])
        path_to_clips = "/".join([path_to_data, "Zwav"])
        metadata_filepaths = {
                split: "/".join([path_to_data, f"{split}"])
                for split in ["txt.done.data.utf8"]
            }
            # (Optional) In non-streaming mode, we can extract the archive locally to have actual local audio files:
        local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else None
        return [
            datasets.SplitGenerator(
                name="txt.done.data.utf8",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["txt.done.data.utf8"],
                    "path_to_clips": path_to_clips,
                },
            ),
        ]

  def _generate_examples(self, local_extracted_archive, archive_iterator, metadata_filepath, path_to_clips):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        pattern = r'\((\s*([^ ]+))\s*"([^"]+)"\s*\)'

        # audio is not a header of the csv files
        data_fields.remove("audio")
        path_idx = data_fields.index("path")
        #directoriesToSkip=["mp3","utt2labels","uttids"]
        all_field_values = {}
        metadata_found = False
        # Here we iterate over all the files within the TAR archive:
        logging.info("Before iterating over the path")
        for path, f in archive_iterator:
            # Parse the metadata CSV file
 #           logging.info('XXXX Inside  archive_iterator and below is path and f')
            logging.info("path {}".format(path))
            logging.info("path_to_clips {}".format(path_to_clips))
            if path == metadata_filepath:
                metadata_found = True
                lines = f.readlines()
                #headline = lines[0].decode("utf-8")
                #column_names = headline.strip().split("\t")
                column_names = ['path', 'sentence']
                assert (
                    column_names == data_fields
                ), f"The file should have {data_fields} as column names, but has {column_names}"
                for line in lines[0:]:
                    match = re.search(pattern, line.decode("utf-8"))
                    if match:
                        code = match.group(2).strip()  # This will contain 'hin_0001'
                        text = match.group(3).strip()  # This will contain the text within the quotes
                        field_values = [code,text]
                    else:
                        print('No match found')
                    #logging.info('split field value {}'.format(field_values))
                    # set full path for mp3 audio file
                    audio_path = "/".join([path_to_clips, field_values[path_idx]])
                    audio_path = ".".join([audio_path,"mp3"])
                    all_field_values[audio_path] = field_values
                    #logging.info('all_field_values:audio_path {}'.format(all_field_values[audio_path]))

            # Else, read the audio file and yield an example
            elif path.startswith(path_to_clips):
                if(path=="iiit_hin/Zmp3/hin_0001.mp3"):
                    logging.info("Came inside path.startswith method with path: {} and all_field_values as{}".format(path,all_field_values))
                assert metadata_found, "Found audio clips before the metadata text file"
                if not all_field_values:
                    logging.info("XXX Break Came inside all_field_values {}".format(all_field_values))
                    break
                if path in all_field_values:
                    # retrieve the metadata corresponding to this audio file

                    field_values = all_field_values[path]
                    if(path=="iiit_hin/Zmp3/hin_0001.mp3"):
                        logging.info('path.startswith(path_to_clips): method  {}'.format(field_values))

                    # if data is incomplete, fill with empty values
                    if len(field_values) < len(data_fields):
                        field_values += (len(data_fields) - len(field_values)) * ["''"]

                    result = {key: value for key, value in zip(data_fields, field_values)}

                    # set audio feature
                    path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                    result["audio"] = {"path": path, "bytes": f.read()}
                    # set path to None if the audio file doesn't exist locally (i.e. in streaming mode)
                    result["path"] = path if local_extracted_archive else None
                    logging.info('path value {}'.format(path))
                    logging.info('result value {}'.format(result))

                    yield path, result

