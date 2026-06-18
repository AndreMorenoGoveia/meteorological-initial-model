========================================================================
Arquivo : data_processed/unified_nc/meteo_unified.nc
Tamanho : 7.4 GB
Formato : NETCDF4
Grupos  : 1
========================================================================

## Grupo: raiz

-- Atributos do grupo
   Conventions: CF-1.8
   title: Unified meteorological dataset (ERA5 + IAG + INMET)
   variables_included: air_temperature_c, dew_point_temperature_c, wind_direction_deg, wind_speed_ms
   sources: ERA5 reanalysis (ECMWF); IAG-USP; INMET

-- Dimensoes (2)
   instance                       18,789
   time                           43,848

-- Variaveis (10)

   time  [int64]  (time)  variavel
      shape  : 43,848
      tamanho: 43,848 celulas
      chunking: contiguous
      attrs  :
         units: seconds since 1970-01-01 00:00:00 UTC
         calendar: proleptic_gregorian
         standard_name: time
      amostra: [1577836800, 1577840400, 1577844000, 1577847600, 1577851200, 1577854800, 1577858400, 1577862000, '...']
      stats  : min=1.57784e+09  max=1.73569e+09  mean=1.65676e+09  NaN=0/43,848 (0.0%)  variavel  [completo]

   latitude  [float32]  (instance)  variavel
      shape  : 18,789
      tamanho: 18,789 celulas
      chunking: [18789]
      filters : {'zlib': True, 'shuffle': True, 'complevel': 4}
      attrs  :
         standard_name: latitude
         units: degrees_north
      amostra: [-21.160999298095703, -21.160999298095703, -21.160999298095703, -21.160999298095703, -21.160999298095703, -21.160999298095703, -21.160999...
      stats  : min=-90  max=90  mean=-3.02083  NaN=0/18,789 (0.0%)  variavel  [completo]

   longitude  [float32]  (instance)  variavel
      shape  : 18,789
      tamanho: 18,789 celulas
      chunking: [18789]
      filters : {'zlib': True, 'shuffle': True, 'complevel': 4}
      attrs  :
         standard_name: longitude
         units: degrees_east
      amostra: [-49.130001068115234, -49.029998779296875, -48.93000030517578, -48.83000183105469, -48.72999954223633, -48.630001068115234, -48.529998779...
      stats  : min=-178  max=180  mean=-5.11963  NaN=0/18,789 (0.0%)  variavel  [completo]

   source_type  [<class 'str'>]  (instance)  variavel
      shape  : 18,789
      tamanho: 18,789 celulas
      chunking: contiguous
      amostra: ['ERA5', 'ERA5', 'ERA5', 'ERA5', 'ERA5', 'ERA5', 'ERA5', 'ERA5', '...']

   instance_id  [<class 'str'>]  (instance)  variavel
      shape  : 18,789
      tamanho: 18,789 celulas
      chunking: contiguous
      amostra: ['era5_inner_lat-21.161_lon-49.130', 'era5_inner_lat-21.161_lon-49.030', 'era5_inner_lat-21.161_lon-48.930', 'era5_inner_lat-21.161_lon-4...

   instance_name  [<class 'str'>]  (instance)  variavel
      shape  : 18,789
      tamanho: 18,789 celulas
      chunking: contiguous
      amostra: ['ERA5 inner (-21.161, -49.130)', 'ERA5 inner (-21.161, -49.030)', 'ERA5 inner (-21.161, -48.930)', 'ERA5 inner (-21.161, -48.830)', 'ERA...

   air_temperature_c  [float32]  (instance, time)  variavel
      shape  : 18,789 x 43,848
      tamanho: 823,860,072 celulas
      chunking: [512, 744]
      filters : {'zlib': True, 'shuffle': True, 'complevel': 4}
      attrs  :
         _FillValue: nan
         standard_name: air_temperature
         units: degree_Celsius
         long_name: Air temperature at 2 m
      amostra: [26.254547119140625, 25.715972900390625, 25.093170166015625, 24.520660400390625, 24.275787353515625, 23.915435791015625, 23.4044494628906...
      stats  : min=-73.2393  max=46.5895  mean=7.6201  NaN=674/195,750 (0.3%)  variavel  [amostra stride=65]

   dew_point_temperature_c  [float32]  (instance, time)  variavel
      shape  : 18,789 x 43,848
      tamanho: 823,860,072 celulas
      chunking: [512, 744]
      filters : {'zlib': True, 'shuffle': True, 'complevel': 4}
      attrs  :
         _FillValue: nan
         standard_name: dew_point_temperature
         units: degree_Celsius
         long_name: Dew point temperature at 2 m
      amostra: [19.563385009765625, 19.647125244140625, 19.732574462890625, 19.373199462890625, 19.570709228515625, 19.886871337890625, 19.9779357910156...    stats  : min=-76.8498  max=29.0386  mean=3.08446  NaN=674/195,750 (0.3%)  variavel  [amostra stride=65]

   wind_direction_deg  [float32]  (instance, time)  variavel
      shape  : 18,789 x 43,848
      tamanho: 823,860,072 celulas
      chunking: [512, 744]
      filters : {'zlib': True, 'shuffle': True, 'complevel': 4}
      attrs  :
         _FillValue: nan
         standard_name: wind_from_direction
         units: degree
         long_name: Wind direction (from, meteorological)
      amostra: [6.587677001953125, 4.6990966796875, 2.044952392578125, 4.776153564453125, 7.58709716796875, 9.876953125, 10.40167236328125, 9.1293334960...
      stats  : min=0.000976562  max=359.999  mean=172.735  NaN=674/195,750 (0.3%)  variavel  [amostra stride=65]

   wind_speed_ms  [float32]  (instance, time)  variavel
      shape  : 18,789 x 43,848
      tamanho: 823,860,072 celulas
      chunking: [512, 744]
      filters : {'zlib': True, 'shuffle': True, 'complevel': 4}
      attrs  :
         _FillValue: nan
         standard_name: wind_speed
         units: m s-1
         long_name: Wind speed at 10 m
      amostra: [2.3664298057556152, 2.425703763961792, 2.7897958755493164, 2.9583325386047363, 2.875267744064331, 2.7566263675689697, 2.8081204891204834...
      stats  : min=0.010596  max=38.6448  mean=5.84742  NaN=674/195,750 (0.3%)  variavel  [amostra stride=65]

## Constantes detectadas
   (nenhuma variavel escalar ou constante numerica detectada)

========================================================================