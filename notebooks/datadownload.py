import datasets
from pathlib import Path
import numpy as np
from datasets.tasks import AutomaticSpeechRecognition
import os
import logging
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub import AudioSegment
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

_DESCRIPTION = "New Common Voice COmbined  dataset"  # Assign a value to _DESCRIPTION

class MusdbDataset(datasets.GeneratorBasedBuilder):

  def _info(self):
      return datasets.DatasetInfo(
          description=_DESCRIPTION,
          features = datasets.Features(
                        {
                            "client_id": datasets.Value("string"),
                            "path": datasets.Value("string"),
                            "audio": datasets.Audio(sampling_rate=48_000),
                            "sentence": datasets.Value("string"),
                            "up_votes": datasets.Value("int64"),
                            "down_votes": datasets.Value("int64"),
                            "age": datasets.Value("string"),
                            "gender": datasets.Value("string"),
                            "accents": datasets.Value("string"),
                            "variant": datasets.Value("string"),
                            "locale": datasets.Value("string"),
                            "segment": datasets.Value("string"),
                        })
      )
  
  
  def _split_generators(self, dl_manager):
        logging.info('XXXX Inside  _split_generators')

        archive_path = dl_manager.download(
            "https://drive.google.com/uc?export=download&id=1_FGBc87W9I8dyyPmH7a2Ou69xV2gBbFq"
            #https://drive.google.com/file/d/1u_ICasXsdv_S3ed42MzUIwxCLHEiSJD5/view?usp=drive_link
            #https://drive.google.com/file/d/1_FGBc87W9I8dyyPmH7a2Ou69xV2gBbFq/view?usp=drive_link
        )

        # First we locate the data using the path within the archive:cv-corpus-15.0-2023-09-08-hi.tar
        path_to_data = "/".join(["cv-corpus-15.0-2023-09-08", "hi"])
        path_to_clips = "/".join([path_to_data, "Zclips"])
        metadata_filepaths = {
                split: "/".join([path_to_data, f"{split}.tsv"])
                for split in ["train", "test", "dev", "other", "validated", "invalidated"]
            }
            # (Optional) In non-streaming mode, we can extract the archive locally to have actual local audio files:
        local_extracted_archive = dl_manager.extract(archive_path) if not dl_manager.is_streaming else None
        print(dl_manager.iter_archive(archive_path))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["train"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["test"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["dev"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="other",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["other"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="validated",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["validated"],
                    "path_to_clips": path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="invalidated",
                gen_kwargs={
                    "local_extracted_archive": local_extracted_archive,
                    "archive_iterator": dl_manager.iter_archive(
                        archive_path
                    ),  # use iter_archive here to access the files in the TAR archives
                    "metadata_filepath": metadata_filepaths["invalidated"],
                    "path_to_clips": path_to_clips,
                },
            ),
        ]

  def _generate_examples(self, local_extracted_archive, archive_iterator, metadata_filepath, path_to_clips):
        """Yields examples."""
        data_fields = list(self._info().features.keys())

        # audio is not a header of the csv files
        data_fields.remove("audio")
        path_idx = data_fields.index("path")

        all_field_values = {}
        metadata_found = False
        # Here we iterate over all the files within the TAR archive:
        print(archive_iterator)
        for path, f in archive_iterator:
            # Parse the metadata CSV file
            logging.info('XXXX Inside  archive_iterator and below is path and f')
            logging.info(path)
            logging.info(type(f))

            if path == metadata_filepath:
                metadata_found = True
                lines = f.readlines()
                headline = lines[0].decode("utf-8")

                column_names = headline.strip().split("\t")
                assert (
                    column_names == data_fields
                ), f"The file should have {data_fields} as column names, but has {column_names}"
                for line in lines[1:]:
                    field_values = line.decode("utf-8").strip().split("\t")
                    # set full path for mp3 audio file
                    audio_path = "/".join([path_to_clips, field_values[path_idx]])
                    
                    all_field_values[audio_path] = field_values
                    logging.info('all_field_values:audio_path {}'.format(all_field_values[audio_path]))

            # Else, read the audio file and yield an example
            elif path.startswith(path_to_clips):
                assert metadata_found, "Found audio clips before the metadata TSV file."
                if not all_field_values:
                    break
                if path in all_field_values:
                    # retrieve the metadata corresponding to this audio file
                    field_values = all_field_values[path]
                    logging.info('path XXX {}'.format(path))
                    logging.info('(path_to_clips XXX){}'.format(path_to_clips))


                    # if data is incomplete, fill with empty values
                    if len(field_values) < len(data_fields):
                        field_values += (len(data_fields) - len(field_values)) * ["''"]

                    result = {key: value for key, value in zip(data_fields, field_values)}

                    # set audio feature
                    path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                    result["audio"] = {"path": path, "bytes": f.read()}

                    # set path to None if the audio file doesn't exist locally (i.e. in streaming mode)
                    result["path"] = path if local_extracted_archive else None

                    yield path, result

