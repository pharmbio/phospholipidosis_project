import typing as tp
from pathlib import Path
import time
import re
import gzip
import gc
import io

from tqdm import tqdm

import pandas as pd
import polars as pl
import numpy as np

import tifffile as tiff
import orjson as json
from sqlalchemy import create_engine
from pydantic import BaseModel, Field

from torch.utils.data import Dataset

PLOT_ALIGNED_IMAGES=bool(0)
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 170)
pd.set_option('display.width', 2000)
pd.set_option("display.max_colwidth", 1000)

project_name='cp-duo'

base_path=Path("/home/jovyan/my projects/cp-duo/data")
# base_path=Path("/share/data/analyses/david/cp-duo/data")

base_path_results = Path('/share/data/cellprofiler/automation/results/')
assert base_path_results.exists()

# set of columns that uniquely identify a location on the plate
loc_keys=["Metadata_AcqID","Metadata_Barcode","Metadata_Well","Metadata_Site"]

snippet_height=244
snippet_width=244

def plot_images(images_cropped:tp.List[tp.Tuple[Path,np.ndarray]],row_index:int):
    # Create a 2x3 grid layout for plotting
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))  # Adjust figsize as needed

    # Plot each image
    for ax, (img_path,image) in zip(axes.flat, images_cropped):
        ax.set_title(img_path.name)
        ax.imshow(image, cmap='gray')  # Display monochrome images with grayscale colormap
        ax.axis('off')  # Hide axes

    # Adjust layout and display the plot
    plt.tight_layout()
    img_save_path=f"cell_snip_{row_index}_2.png"
    plt.savefig(img_save_path)
    plt.close()

    # for debugging
    # print(f"created and saved fig in {time.time()-start_time}s")

    print(f"saved to {img_save_path}")

class FeatureFile(BaseModel):
    name:str
    frame:pd.DataFrame

    class Config:
        arbitrary_types_allowed=True

tiff_imread_timer=[0.0]*2

image_cache:tp.Dict[Path,np.ndarray]={}
def read_image(path:Path,cache:bool=True,read_from_cache_if_present:bool=True)->np.ndarray:
    """
    read image from disk

    also buffers images to avoid reading from our slow fs
    """

    global tiff

    if not (read_from_cache_if_present and path in image_cache):
        if 0:
            img=tiff.imread(path)
        else:
            # measure time to read data from disk
            start_time=time.time()
            with path.open('rb') as f:
                tiff_data = f.read()
            tiff_io = io.BytesIO(tiff_data)
            tiff_imread_timer[0]+=time.time()-start_time

            # measure time to decompress data in ram
            start_time=time.time()
            with tiff.TiffFile(tiff_io) as file:
                img = file.asarray()
            tiff_imread_timer[1]+=time.time()-start_time

        if cache:
            image_cache[path]=img

        return img
    
    return image_cache[path]

def align_image(img:tp.Union[np.ndarray,str,Path],dx:int,dy:int,is_bf:bool)->np.ndarray:
    """
    apply the image alignment

    crops the image based on its type (is_bf flag) and dx and dy values

    this code handles the direction of the alignment as well
    """

    if dy>0:
        if is_bf:
            # shift brightfield by dy,dx to align with fluorescences
            slice_y=slice(None,-dy)
        else:
            # i.e. crop fluorescence by those amounts
            slice_y=slice(dy,None)
    else:
        if is_bf:
            slice_y=slice(dy,None)
        else:
            slice_y=slice(None,-dy)

    if dx>0:
        if is_bf:
            slice_x=slice(None,-dx)
        else:
            slice_x=slice(dx,None)
    else:
        if is_bf:
            slice_x=slice(dx,None)
        else:
            slice_x=slice(None,-dx)

    if isinstance(img,(Path,str)):
        img=read_image(Path(img),cache=False)
    
    return img[slice_y,slice_x]

class PlateData(BaseModel):
    """ plate_id is also barcode """
    plate_id:str
    acquisition_id_fluo:str
    acquisition_id_brightfield:str
    path_fluo:str
    path_brightfield:str

    pipeline_id_segmentation:str
    pipeline_id_features:str
    pipeline_id_qc:str
    layout_name:str

    cell_line:str
    z_index:int

    class Config:
        arbitrary_types_allowed=True

    @property
    def full_fluo_path(self)->Path:
        base_image_path=Path(f"/share/mikro2/squid/{project_name}")

        full_fluo_path=Path(base_image_path/self.path_fluo)
        return full_fluo_path

    @property
    def full_brightfield_path(self)->Path:
        base_image_path=Path(f"/share/mikro2/squid/{project_name}")

        full_brightfield_path=Path(base_image_path/self.path_brightfield)
        return full_brightfield_path
    
    @property
    def full_features_path(self)->Path:
        full_features_path=base_path_results/f"{self.plate_id}/{self.acquisition_id_fluo}/{self.pipeline_id_features}"
        return full_features_path
    
    @property
    def full_qc_path(self)->Path:
        full_qc_path=base_path_results/f"{self.plate_id}/{self.acquisition_id_fluo}/{self.pipeline_id_qc}"
        return full_qc_path
        
    @property
    def all_feature_paths(self)->tp.Dict[str,Path]:
        """
        key is feature name, value is path to respective file

        some notable columns in each file:
        - nuclei: Metadata_AcqID, Metadata_Barcode, Metadata_Well, Metadata_Site, ObjectNumber
        - cytoplasm: Metadata_AcqID, Metadata_Barcode, Metadata_Well, Metadata_Site, ObjectNumber
        - cells: Metadata_AcqID, Metadata_Barcode, Metadata_Well, Metadata_Site, ObjectNumber

        notes:
        - 'cells' dataframe is the root for the final dataframe. ObjectNumber here identifies a cell
        - nuclei and cytoplasm both have ObjectNumber entries, which do NOT refer to the cell they are in! \
        instead, Parent_cells refers to the cell in which they are contained.
        """

        return {
            feature_name:self.full_features_path/("featICF_"+feature_name+".parquet")
            for feature_name
            in ["cells","cytoplasm","nuclei"]
        }
        
    def check_paths_valid(self):
        assert self.full_fluo_path.exists(), f"{self.full_fluo_path} does not exist"
        assert self.full_brightfield_path.exists(), f"{self.full_brightfield_path} does not exist"
        assert self.full_features_path.exists()
        for feature_name,feature_file in self.all_feature_paths.items():
            assert feature_file.exists(), f"feature {feature_name} file {feature_file} does not exist"
        
        #path_mask = base_path_results/f'{self.plate_id}/{self.acquisition_id_fluo}/{self.pipeline_id_segmentation}/masks'

    # allow lookup of location identifiers in dictionaries
    def make_location_key(self,well:str,site:tp.Union[int,str])->str:
        return f"{self.plate_id};{well};{site}"

    # mask key extends location identifier with cell id
    def make_mask_key(self,well:str,site:tp.Union[int,str],cell_id:str)->str:
        return self.make_location_key(well,site)+f";{cell_id}"

    mask_paths:tp.Dict[str,Path]=Field(default_factory=dict)
    def get_mask(self,well:str,site:int,cell_id:tp.Union[int,str])->tp.Optional[Path]:
        """
        get binary segmentation mask just for this cell

        the mask should (?) just be the same size as the bounding box of the cell

        cell_id as seen in the feature dataframe
        """
        cell_id=str(cell_id)

        key=self.make_mask_key(well,site,cell_id)

        return self.mask_paths.get(key)

    # this is assigned differently for each plate
    alignment_dataframe:pd.DataFrame=Field(default_factory=pd.DataFrame)
    alignment_buffer:tp.Dict[str,tp.Tuple[int,int]]=Field(default_factory=dict)
    def get_alignment_info(self,well:str,site:int)->tp.Tuple[int,int]:
        """
        fetching the alignment info from the alignment_info dataframe takes up to 50ms, so
        we cache it here

        returns offset (dx,dy), where the brightfield is shifted by that amount from fluo
        """
        
        key=self.make_location_key(well,site)
        if key in self.alignment_buffer:
            return self.alignment_buffer[key]

        # this alignmen info section takes up to 50ms
        alignment_info=self.alignment_dataframe[
            (self.alignment_dataframe["plate"]==self.plate_id) \
            & (self.alignment_dataframe["well"]==well) \
            & (self.alignment_dataframe["site"]==site)
        ]

        assert len(alignment_info)==1, f"not one alignment info found {alignment_info}"
        alignment_info_dict=alignment_info.iloc[0].to_dict()

        dx=alignment_info_dict["dx"]
        dy=alignment_info_dict["dy"]

        self.alignment_buffer[key]=(dx,dy)

        return dx,dy
    
    image_file_buffer_fluo:tp.Dict[str,tp.List[Path]]=Field(default_factory=dict)
    image_file_buffer_bf:tp.Dict[str,tp.List[Path]]=Field(default_factory=dict)
    def get_image_files_for_plate(self,bf:bool)->tp.List[Path]:
        """ our fs is quite slow, so this function buffers glob results """

        if bf:
            if not self.plate_id in self.image_file_buffer_bf:
                bf_image_files=list(self.full_brightfield_path.glob(f"*.tif*"))
                self.image_file_buffer_bf[self.plate_id]=bf_image_files
                return bf_image_files
            
            return self.image_file_buffer_bf[self.plate_id]
        else:
            if not self.plate_id in self.image_file_buffer_fluo:
                fluo_image_files=list(self.full_fluo_path.glob(f"*.tif*"))
                self.image_file_buffer_fluo[self.plate_id]=fluo_image_files
                return fluo_image_files
            
            return self.image_file_buffer_fluo[self.plate_id]

    aligned_image_buffer:tp.Dict[str,tp.List[tp.Tuple[Path,np.ndarray]]]={}
    def get_images(self,row,well:str,site:int)->tp.List[tp.Tuple[Path,np.ndarray]]:
        """
        get 6 channel images for the target location on the plate
        """

        key=self.make_location_key(well,site)
        if key in self.aligned_image_buffer:
            return self.aligned_image_buffer[key]

        # if images are not already in buffer, old images are likely unused from now on, so delete them
        self.aligned_image_buffer={}
        gc.collect()

        dx,dy=self.get_alignment_info(well,site)

        # tuple of (full image path, image data)
        images:tp.List[tp.Tuple[Path,np.ndarray]]=[]

        # fluorescence channels first
        for channelname in [
            "Fluorescence_405_nm_Ex",
            "Fluorescence_488_nm_Ex",
            "Fluorescence_561_nm_Ex",
            "Fluorescence_638_nm_Ex",
            "Fluorescence_730_nm_Ex",
        ]:
            fluo_image_files=self.get_image_files_for_plate(bf=False)
            fluo_image_files=[
                f for f in fluo_image_files
                if re.match(rf"{row.Metadata_Well}_s{row.Metadata_Site}_.*_{channelname}\.tif.*",f.name)
            ]

            assert len(fluo_image_files)==1, f"found not one fluo image {fluo_image_files}"
            images.append((
                fluo_image_files[0],
                align_image(
                    read_image(fluo_image_files[0]),
                    dx=dx,dy=dy,
                    is_bf=False,
                )
            ))

        # then brightfield channel[s]
        for channelname in [
            # single brightfield channel
            "BF_LED_matrix_full",
        ]:
            bf_image_files=self.get_image_files_for_plate(bf=True)
            bf_image_files=[
                f for f in bf_image_files
                # note the _z2 component in the filename (we have z=3 stacks)
                if re.match(rf"{row.Metadata_Well}_s{row.Metadata_Site}_.*_z2_{channelname}\.tif.*",f.name)
            ]

            assert len(bf_image_files)==1, f"found not one brightfield image {bf_image_files}"
            images.append((
                bf_image_files[0],
                align_image(
                    read_image(bf_image_files[0]),
                    dx=dx,dy=dy,
                    is_bf=True,
                )
            ))

        self.aligned_image_buffer[key]=images

        return images

times:tp.List[float]=[0.0]*5

class CombiningCellLoader(Dataset):
    """
    combines data from multiple sources into the target format

    for several reasons, this structure can only iterate over the results, and not
    provide random access to any cell

    this structure iterates over multiple plates sequentially, so its total length is also not known in advance
    """

    def __init__(self,perform_qc:bool=True):
        self.perform_qc=perform_qc

    def __iter__(self): # -> tp.Generator[tp.Tuple[PlateData, tp.Dict[str, tp.Any], tp.List[tp.Tuple[Path,np.ndarray]]]]:
        global times, plt, np, image_cache, PLOT_ALIGNED_IMAGES

        assert (base_path/"alignment.parquet").exists()
        alignment_dataframe=pd.read_parquet(base_path/"alignment.parquet")

        metadata_layout_path = base_path/"20241011-cp-duo-metadata-SARTORIUS.csv"
        assert metadata_layout_path.exists()
        metadata_layout=pd.read_csv(metadata_layout_path)

        # rename layout ids to match those in plates.csv
        metadata_layout["layout_id"]=[l.split("-")[-1] for l in metadata_layout["layout_id"]]
        metadata_layout=metadata_layout.rename(columns={
            "barcode":"sartorius_barcode",
            "well_id":"Metadata_Well",
        })
        # there are multiple plates with the same layout.
        # we only care about the layout, not the (plate,layout) combination,
        # so we just take the first of each.
        metadata_layout=metadata_layout.groupby(["Metadata_Well","layout_id"]).first()
        # print("metadata:\n",metadata_layout.head(2))

        # in plates.csv:
        # acq_id_cp is the CP acquisition ID
        # acq_id_bf is the brightfield acq ID under automation/results
        # path_cp is the path to the CP images
        # path_bf is the path to the BF images under mikrp2/squid/cp-duo
        # also 'seg_id', 'qc_id', 'feat_id', 'cell_line'
        # layout_id is the name of the layout (multiple plates may have the same layout). e.g. L1, and in the metadata*.csv its cpduo*-*-L01

        plates_df = pd.read_csv(base_path/'plates.csv', sep=';')
        #print("plates_df:")
        #print(plates_df.head(2))

        for plate_id in tqdm(plates_df['plate_id'].unique(),desc="plate"):
            # empty all caches
            image_cache={}

            # ensure garbage is collected (old objects may hold on to large amounts of memory, especially dataframes)
            gc.collect()

            # -- actually start new iteration

            plate_df = plates_df[plates_df['plate_id']==plate_id]

            # some plates are not yet segmented, so their seg_id is nan
            # -> conversion to int throws -> skip those plates
            try:
                # this column should be int-like, but is actually a float, which we \
                # fix by converting float->int->str
                pipeline_id_segmentation=str(int(plate_df['seg_id'].values[0]))
            except:
                continue

            plate=PlateData(
                plate_id = plate_df["plate_id"].values[0],

                acquisition_id_fluo = str(plate_df['acq_id_cp'].values[0]),
                acquisition_id_brightfield = str(plate_df["acq_id_bf"].values[0]),
                path_fluo = str(plate_df['path_cp'].values[0]),
                path_brightfield = plate_df["path_bf"].values[0],
                pipeline_id_features = str(plate_df['feat_id'].values[0]),
                pipeline_id_qc = str(plate_df['qc_id'].values[0]),

                pipeline_id_segmentation = pipeline_id_segmentation,

                layout_name=plate_df["layout_id"].values[0],
                cell_line=plate_df["cell_line"].values[0],
                z_index = 1,
            )
            plate.check_paths_valid()

            plate.alignment_dataframe=alignment_dataframe

            # there may be other qc files, e.g. qcRAW_nuclei, but those do not use the same cell identifiers
            # that are used in the feature extraction pipelines, so they are nearly 100% useless.
            # but we can use the image qc to exclude bad images
            qc_dataframe:tp.Optional[pd.DataFrame]=None
            if self.perform_qc:
                for qc_file in plate.full_qc_path.glob("qcRAW_images*.parquet"):
                    print(f"found image qc file {qc_file = }")

                    qc_dataframe=pd.read_parquet(qc_file)

                    # remove unused columns
                    exclude_columns=set()
                    for col in qc_dataframe.columns:
                        if "ImageNumber" in col: exclude_columns.add(col)
                        if "FileName" in col: exclude_columns.add(col)
                        if "PathName" in col: exclude_columns.add(col)
                        if "ExecutionTime" in col: exclude_columns.add(col)
                        if "URL" in col: exclude_columns.add(col)
                        if "Height" in col: exclude_columns.add(col)
                        if "Width" in col: exclude_columns.add(col)
                        if "MD5Digest" in col: exclude_columns.add(col)
                    qc_dataframe.drop(columns=list(exclude_columns),inplace=True)

                    # use the qc flags, which in theory is the way to go.
                    # in practice, 99% of images are flagged that way, so
                    # we need to do something else.
                    if False:
                        flags=None
                        for col in qc_dataframe.columns:
                            if not col.startswith("qc_flag_"):
                                continue

                            if flags is None:
                                flags=qc_dataframe[col]==1
                            else:
                                flags|=qc_dataframe[col]==1
                        print(f"qc flagged: {len(qc_dataframe[flags])} out of {len(qc_dataframe)}")

                    if True:
                        unused_columns=set()
                        for col in qc_dataframe.columns:
                            if not (col.startswith("ImageQuality_") or col.startswith("Metadata_")):
                                unused_columns.add(col)

                            # list from petters code
                            for name in [
                                'TotalArea', 'Scaling', 'TotalIntensity', 'Correlation', 'PercentMinimal',
                                'LocalFocusScore', 'MinIntensity', 'MedianIntensity', 'MADIntensity',
                                'ThresholdMoG', 'ThresholdBackground', 'ThresholdKapur',
                                'ThresholdMCT', 'ThresholdOtsu', 'ThresholdRidlerCalvard', 'ThresholdRobustBackground',
                                'PercentMaximal'
                            ]:
                                if name in col:
                                    unused_columns.add(col)

                        qc_dataframe.drop(columns=list(unused_columns),inplace=True)
                        qc_dataframe.rename({c:c.split("_",maxsplit=1)[1] for c in qc_dataframe.columns},inplace=True)

                        # the remaining features are FocusScore, MeanIntensity, StdIntensity, PowerLogLogSlope, MaxIntensity

                        # flag and remove sites with values outside the mean value range
                        flags=None
                        threshold_num_stds=5
                        for col in qc_dataframe.columns:
                            if col.startswith("Metadata"):
                                continue

                            target_col=qc_dataframe[col]
                            scaled_col=(target_col-target_col.mean())/target_col.std()
                            if flags is None:
                                flags=(scaled_col>threshold_num_stds) | (scaled_col < (-threshold_num_stds))
                            else:
                                flags|=(scaled_col>threshold_num_stds) | (scaled_col < (-threshold_num_stds))

                        print(f"flagged {len(qc_dataframe[flags])} out of {len(qc_dataframe)}")

                        # actually remove the flagged rows
                        assert flags is not None
                        qc_dataframe=qc_dataframe[~flags]

                        qc_dataframe.rename(columns={
                            c:"QCFLAG_"+c for
                            c in qc_dataframe.columns
                            if not col.startswith("Metadata")},
                            inplace=True,
                        )

            print(f"{plate=}")

            # currently found in another data source (plates.csv)
            if bool(0):
                db_uri = 'postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb'
                result_type:tp.Literal["cp-qc","cp-features"]="cp-features"
                query = f"""
                    SELECT plate_barcode,plate_acq_name,plate_acq_id,analysis_id,meta,results
                    FROM image_analyses_per_plate
                    WHERE project LIKE '{project_name}%%'
                    AND meta->>'type' = '{result_type}'
                    AND plate_barcode = '{plate.plate_id}'
                    AND analysis_date IS NOT NULL
                    and analysis_error is NULL
                    ORDER BY plate_barcode 
                """
                engine = create_engine(db_uri)
                with engine.connect() as connection:
                    project_info = pd.read_sql(query, connection)
                    print("features")
                    print(project_info.head(2))

                db_uri = 'postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb'
                result_type:tp.Literal["cp-qc","cp-features"]="cp-qc"
                query = f"""
                    SELECT plate_barcode,plate_acq_name,plate_acq_id,analysis_id,meta,results
                    FROM image_analyses_per_plate
                    WHERE project LIKE '{project_name}%%'
                    AND meta->>'type' = '{result_type}'
                    AND plate_barcode = '{plate.plate_id}'
                    AND analysis_date IS NOT NULL
                    and analysis_error is NULL
                    ORDER BY plate_barcode 
                """
                engine = create_engine(db_uri)
                with engine.connect() as connection:
                    project_info = pd.read_sql(query, connection)
                    print("qc")
                    print(project_info.head(2))

                return

            # use polars to read and merge feature dataframes

            start_time=time.time()

            feature_dataframes:tp.Dict[str,pl.DataFrame]={}

            for feature_name,feature_file in plate.all_feature_paths.items():
                # Read the Parquet file into a Polars DataFrame
                feature_dataframe = pl.read_parquet(feature_file)

                # Identify columns to remove
                remove_columns = [
                    col for col in feature_dataframe.columns
                    if col.startswith("FileName") \
                        or col.startswith("PathName") \
                        or col.startswith("ImageNumber")
                ]

                # Drop the identified columns
                feature_dataframe = feature_dataframe.drop(remove_columns)

                # Fill null values with 0
                feature_dataframe = feature_dataframe.fill_null(0)

                # Assign the resulting Polars DataFrame back to the dictionary
                feature_dataframes[feature_name] = feature_dataframe

            print(f"done reading feature files after {time.time()-start_time}s")

            assert "ObjectNumber" in feature_dataframes["cells"].columns
            assert "Parent_cells" in feature_dataframes["nuclei"].columns
            assert "Parent_cells" in feature_dataframes["cytoplasm"].columns

            loc_on=loc_keys+["cell_id"]

            start_time=time.time()

            cells_df = (feature_dataframes["cells"]).rename({"ObjectNumber": "cell_id"})
            nuclei_df = (feature_dataframes["nuclei"]).rename({"Parent_cells": "cell_id"})
            cytoplasm_df = (feature_dataframes["cytoplasm"]).rename({"Parent_cells": "cell_id"})

            # Rename columns to avoid collisions
            cells_df = cells_df.rename({c: f"{c}_DFcells" for c in set(cells_df.columns) - set(loc_on)})
            nuclei_df = nuclei_df.rename({c: f"{c}_DFnuclei" for c in set(nuclei_df.columns) - set(loc_on)})
            cytoplasm_df = cytoplasm_df.rename({c: f"{c}_DFcytoplasm" for c in set(cytoplasm_df.columns) - set(loc_on)})

            # Perform inner joins
            complete_dataframe = (
                cells_df.join(nuclei_df, on=loc_on, how="inner")
                        .join(cytoplasm_df, on=loc_on, how="inner")
            )

            # convert to pandas for later operations
            complete_dataframe=complete_dataframe.to_pandas()
            
            print(f"merging took {(time.time()-start_time):.2f}s")

            if qc_dataframe is not None:
                start_time=time.time()
                complete_dataframe=complete_dataframe.merge(
                qc_dataframe,
                    how="inner",
                    left_on=loc_keys,
                    right_on=loc_keys,
                )
                complete_dataframe.drop(columns=[
                    c for c
                    in complete_dataframe.columns
                    if c.startswith("QCFLAG_")
                ],inplace=True)
                print(f"filtered out flagged images in {time.time()-start_time}s")

            # for debugging
            # print("dataframe cols:",*[c for c in complete_dataframe.columns if not c.startswith("RadialDistribution")],sep="\n")

            # then check for cells with more than one nucleus (or cytoplasm):
            dataframe_mask_cellsWithMoreThanOneCytoplasm=complete_dataframe["Children_cytoplasm_Count_DFcells"]>1
            dataframe_mask_cellsWithMoreThanOneNucleus=complete_dataframe["Children_nuclei_Count_DFcells"]>1

            print(f"num cells (supposed): {len(complete_dataframe)}, of which"\
                f" {len(complete_dataframe[dataframe_mask_cellsWithMoreThanOneCytoplasm])} with more"\
                f" than one cytoplasm, and {len(complete_dataframe[dataframe_mask_cellsWithMoreThanOneNucleus])}"\
                f" with more than one nucleus")
            
            print(f"len before dealing with problematic cells {len(complete_dataframe)}")

            integer_columns:tp.Dict[str,str]={}
            for col in complete_dataframe.columns:
                if complete_dataframe[col].dtype=="int32":
                    integer_columns[col]="int32"
                if complete_dataframe[col].dtype=="int64":
                    integer_columns[col]="int64"

            print("-- before grouping")
            for col in complete_dataframe.columns:
                if complete_dataframe[col].isna().any():
                    print(f"found nan in {col}")
                if (complete_dataframe[col]==float("nan")).any():
                    print(f"found inf in {col}")
            print("checked for nan and inf")

            dataframe_mask_cellsWithDuplicatedStructures = dataframe_mask_cellsWithMoreThanOneCytoplasm | dataframe_mask_cellsWithMoreThanOneNucleus
            if bool(1):
                affected_rows = complete_dataframe[dataframe_mask_cellsWithDuplicatedStructures]
                unaffected_rows = complete_dataframe[~dataframe_mask_cellsWithDuplicatedStructures]

                # Group affected rows by `loc_on` and calculate mean for each group
                resolved_rows = affected_rows.groupby(loc_on).mean().reset_index()

                # Combine the resolved rows back with unaffected rows
                complete_dataframe = pd.concat([unaffected_rows, resolved_rows], ignore_index=True)
            else:
                # remove them
                complete_dataframe=complete_dataframe[~(dataframe_mask_cellsWithMoreThanOneCytoplasm|dataframe_mask_cellsWithMoreThanOneNucleus)]

            print("-- after grouping")
            for col in complete_dataframe.columns:
                if complete_dataframe[col].isna().any():
                    print(f"found nan in {col}")
                if (complete_dataframe[col]==float("nan")).any():
                    print(f"found inf in {col}")
            print("checked for nan and inf")

            complete_dataframe=complete_dataframe.astype(integer_columns)
            print("ensured integer columns are integers after grouping")

            # problematic, meaning: cells with more than one nucleus
            print(f"len after dealing with problematic cells {len(complete_dataframe)}")

            print("adding compound information")
            start_time=time.time()

            print(f"before join with plates.csv {len(complete_dataframe)=}")

            # join with plates.csv to get layout id for each plate
            complete_dataframe=complete_dataframe.merge(
                plates_df.drop(columns=[c for c in plates_df.columns if not c in set(["plate_id","layout_id"])]),
                how="inner",
                left_on=["Metadata_Barcode"],
                right_on=["plate_id"],
            ).drop(columns=[
                # duplicate information with Metadata_Barcode
                "plate_id",
            ])
            print(f"after join with plates.csv {len(complete_dataframe)=}")
            # then join with sartorius table to get compound information
            complete_dataframe=complete_dataframe.merge(
                metadata_layout.drop(columns=[
                    # external information, not relevant to us

                    "seeded", # seed date
                    "painted", # paint date
                    "painted_type", # painting protocol
                    "stock_conc",
                    "stock_conc_unit",
                    "cmpd_vol", # cmpd_conc probably better choice to take further
                    "cmpd_vol_unit",
                    "well_vol",
                    "well_vol_unit",
                    "plate_size", # number of wells
                    "plate_type", # plate model name
                    "treatment", # e.g. 48
                    "treatment_units", # e.g. 'h' (for hours)
                    "cells_per_well", # we have our own numbers for this
                ]),
                how="inner",
                left_on=["Metadata_Well","layout_id"],
                right_on=["Metadata_Well","layout_id"],
            )

            # rename columns to indicate metadata

            metadata_colnames=[]
            for key in complete_dataframe.columns:
                if "Zernike" in key: continue
                # Radius, Radial
                if "Radi" in key: continue
                if "Location" in key: continue
                if "Intensity" in key: continue
                if "Granularity" in key: continue
                if "Correlation" in key: continue
                if "AreaShape" in key: continue
                if "Neighbors" in key: continue
                if "Parent" in key: continue
                if "Object_Number" in key: continue
                if "ObjectNumber" in key: continue
                # Children_<x>_Count
                if "Children" in key: continue

                # do not duplicate annotation
                if key.startswith("Metadata_"):continue
                metadata_colnames.append(key)

            complete_dataframe=complete_dataframe.rename(columns={c:"Metadata_"+c for c in metadata_colnames})

            print(f"after join with compound information {len(complete_dataframe)=}")

            print(f"added compound information in {time.time()-start_time} s")

            # sort to improve image locality
            start_time=time.time()
            complete_dataframe=complete_dataframe.sort_values(by=["Metadata_Well","Metadata_Site"])
            print(f"sorted in {time.time()-start_time}")

            # mask existence is essentially a sparse feature, so we pre-calculate it in advance
            # which removes large amounts of redundant filesystem interactions
            try:
                mask_pipeline_id=plate.pipeline_id_segmentation
                
                # path containing directories for each site
                all_mask_path=f"/share/data/cellprofiler/automation/results/{plate_id}/"\
                                    f"{plate.acquisition_id_fluo}/{mask_pipeline_id}/masks"
                
                # for debugging
                # print(f"""{all_mask_path=} ; {Path(all_mask_path).exists()=} ; {f"{well}_s{site}_*_Fluorescence_561_nm_Ex"}""")
                
                # for each site on the plate
                for dir in Path(all_mask_path).iterdir():
                    # check for dir=<plate>_s<site>_*_<channelname>.
                    # deconstruct it to retrieve metadata
                    [p_well,p_site,_]=dir.name.split("_",maxsplit=2)
                    # strip "s" prefix of actual site index
                    p_site=p_site[1:]

                    # mask name is <image name>_<cell_id>.tif[f]
                    # cell_id as seen in the feature dataframe
                    for mask_name in dir.iterdir():
                        mask_cell_id=mask_name.stem.split("_")[-1]

                        plate.mask_paths[plate.make_mask_key(p_well,p_site,mask_cell_id)]=mask_name

            except Exception as e:
                print(f"error - {e}")

            # iterate over all cells on the plate
            for row_index,row in tqdm(
                enumerate(complete_dataframe.itertuples()),
                total=len(complete_dataframe)
            ):
                # during development, most sites did not have masks available, so we check for masks first
                # to avoid discarding the work on image alignment
                start_time=time.time()
                cell_mask_path=plate.get_mask(row.Metadata_Well,row.Metadata_Site,row.Metadata_cell_id)
                times[3]+=time.time()-start_time
                if cell_mask_path is None:
                    # for debugging
                    # print(f"no mask for {plate=} {row.Metadata_Well=} {row.Metadata_Site}")
                    continue

                # row is Namedtuple corresponding to the dataframe rows
                # crop coordinates:
                # bound box has 4 components: xmin, xmax, ymin, ymax
                cell_bb_x0=row.AreaShape_BoundingBoxMinimum_X_DFcells
                cell_bb_x1=row.AreaShape_BoundingBoxMaximum_X_DFcells
                cell_bb_y0=row.AreaShape_BoundingBoxMinimum_Y_DFcells
                cell_bb_y1=row.AreaShape_BoundingBoxMaximum_Y_DFcells
                # center of bounding box (actually not 100% sure, but very likely thats what these are)
                cell_center_x=row.AreaShape_Center_X_DFcells
                cell_center_y=row.AreaShape_Center_Y_DFcells

                # get alignment info for this site
                start_time=time.time()
                dx,dy=plate.get_alignment_info(row.Metadata_Well,row.Metadata_Site)
                times[0]+=time.time()-start_time

                # adjust coordinates
                cell_bb_x0-=dx
                cell_bb_x1-=dx
                cell_bb_y0-=dy
                cell_bb_y1-=dy
                cell_center_x-=dx
                cell_center_y-=dy

                start_time=time.time()
                images:tp.List[tp.Tuple[Path,np.ndarray]]=plate.get_images(row,row.Metadata_Well,row.Metadata_Site)
                times[1]+=time.time()-start_time

                # center crop cell, based on bounding box outline
                bb_width=cell_bb_x1-cell_bb_x0
                bb_height=cell_bb_y1-cell_bb_y0

                center_x=int(cell_bb_x0+((bb_width)/2))
                center_y=int(cell_bb_y0+((bb_height)/2))

                # exclude cells that are larger than snippet size
                if bb_width > snippet_width:
                    continue
                if bb_height > snippet_height:
                    continue

                # crop to uniform snippet size
                cell_crop_x0=int(center_x-snippet_width/2)
                cell_crop_x1=int(center_x+snippet_width/2)
                cell_crop_y0=int(center_y-snippet_height/2)
                cell_crop_y1=int(center_y+snippet_height/2)

                # skip cells that are outside the image after alignment
                if cell_crop_x0<0:
                    continue
                if cell_crop_y0<0:
                    continue
                if cell_crop_x1>images[0][1].shape[1]:
                    continue
                if cell_crop_y1>images[0][1].shape[0]:
                    continue

                # for debugging
                # print(f"{cell_crop_x0=} {cell_crop_x1=} {cell_crop_x1=} {cell_crop_y1=} ({images[0][1].shape=})")

                # cropping takes a fraction of a ms
                start_time=time.time()
                images_cropped=[
                    (path,image[
                        cell_crop_y0:cell_crop_y1,
                        cell_crop_x0:cell_crop_x1,
                    ])
                    for (path,image)
                    in images
                ]
                times[2]+=time.time()-start_time

                start_time=time.time()
                images_cropped+=[(
                    cell_mask_path,

                    align_image(
                        cell_mask_path,
                        dx=dx,dy=dy,
                        is_bf=False,
                    )[
                        cell_crop_y0:cell_crop_y1,
                        cell_crop_x0:cell_crop_x1,
                    ],
                )]
                times[4]+=time.time()-start_time

                # plot aligned channels
                if PLOT_ALIGNED_IMAGES:
                    plot_images(images_cropped,row_index=row_index)

                row_dict:tp.Dict[str,tp.Any]=row._asdict() # type: ignore
                yield plate,row_dict,images_cropped

BASE_CROP_PATH=Path("/home/jovyan/my projects/cp-duo/crop_generation/crops")
assert BASE_CROP_PATH.exists()

class PreprocessedCellLoader(Dataset):
    def __init__(self):
        self.cell_list=list(BASE_CROP_PATH.glob("*.npz"))
        print(f"found {len(self)} cells")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self,idx:int)->tp.Tuple[tp.Dict[str,tp.Any],np.ndarray]:
        data=np.load(self.cell_list[idx],allow_pickle=True)
        images:np.ndarray=data["images"]

        metadata_filepath=self.cell_list[idx].with_suffix(".json.gz")
        with gzip.open(metadata_filepath,mode="rb") as metadata_file:
            # Read the bytes, decode to a string, then load the JSON
            metadata = json.loads(metadata_file.read()) # Load JSON from the decoded string

        return metadata,images

    def __len__(self)->int:
        # we have 2 files per cell, but only the npz file is contained in this list
        return len(self.cell_list)

def main():
    # UnboundLocal error if this is not present
    global plt, tiff, processed_time

    # flag to compress data to disk for all crops in the CombiningCellLoader
    # if false, just combines and iterates over the data
    # if true, combines and writes combinations to disk during iteration
    COMPRESS_DATA=False

    cells_found=0
    start_time=time.time()
    for plate,row,images_cropped in CombiningCellLoader(perform_qc=False):
        cells_found+=1

        if COMPRESS_DATA:
            cell_name=plate.make_mask_key(
                well=row["Metadata_Well"],site=row["Metadata_Site"],
                cell_id=row["Metadata_cell_id"]
            )

            npz_path=BASE_CROP_PATH/(cell_name+".npz")
            assert not npz_path.exists(), f"{npz_path=}"
            metadata_filepath=BASE_CROP_PATH/(cell_name+".json.gz")
            assert not metadata_filepath.exists(), f"{metadata_filepath=}"

            # numbers for one test cell:
            # 388kB for the images (i.e. saves about 50%)

            # TODO try separating the mask channel from the 6 (5 fluo + 1 bf) image channels
            # the mask is binary, the images are 16(12) bit deep

            np.savez_compressed(npz_path,images=np.array([img for (img_path,img) in images_cropped]))

            with gzip.open(metadata_filepath,mode="wb+") as metadata_file:
                metadata_file.write(json.dumps(row)) # type:ignore

    # code for debugging:
    # print code timings (used to measure performance of certain code sections)
    print("\n".join([f"{i}: {t}s" for i,t in enumerate(times)]))

    print(f"iteratively found {cells_found} cells in {time.time()-start_time}s")

    cells_found=0
    start_time=time.time()
    for row,images in tqdm(PreprocessedCellLoader()):
        cells_found+=1
        #print(f"found data: {row['Metadata_Barcode'],row['Metadata_Well'],row['Metadata_Site'],row['cell_id']} {images.shape=}")
        
    print(f"randomly found {cells_found} cells in {time.time()-start_time}s")

if __name__=="__main__":
    main()

    print(f"tiff {tiff_imread_timer[0]=}")
    print(f"tiff {tiff_imread_timer[1]=}")