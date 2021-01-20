import matplotlib
matplotlib.use('Agg')  # Tell matplotlib not to use the x-window to generate plots
import matplotlib.pyplot as plt
import numpy as np
from fermipy.gtanalysis import GTAnalysis
from fermipy.plotting import ROIPlotter
from scipy.optimize import curve_fit
import argparse
import yaml
import os
import shutil
import subprocess


class fermiAnal:

    def __init__(self):
        self.gta = None
        self.titleSuffix = ''
        self.target = ''
        self.targetShort = ''

        return

    # fitSource, redoSetup
    def analyze(self, configFile, analFile):

        # This is here only because at the moment FermiPy fails if ccube.fits exists
        # It is fixed in the latest development version of FermiPy, but in the meantime
        # need to do this
        with open(configFile, 'r') as fConfig:
            tempConfig = yaml.safe_load(fConfig)
        if os.path.exists(tempConfig['fileio']['outdir'] + '/ccube.fits'):
            os.remove(tempConfig['fileio']['outdir'] + '/ccube.fits')

        with open(analFile, 'r') as fAnal:
            analCofing = yaml.safe_load(fAnal)

        self.gta = GTAnalysis(configFile, logging={'verbosity': analCofing['verbosity']})

        # Access to the actual configuration dictionary is via
        # self.gta._config['model']['catalogs'][0]

        target = ''
        for allItems in self.gta.config.items():
            for section in allItems:
                if isinstance(section, dict):
                    for key, value in section.items():
                        if key == 'target':
                            target = str(value)

        self.target = target
        self.targetShort = self.gta.outdir.split('/')[-1]

        self.titleSuffix = analCofing['title'].replace(" ", "").split("+")[0].split(".")[0]

        if analCofing['doFit']:
            self.performFit(analCofing['redoSetup'])
        if analCofing['plotMaps']:
            self.plotMaps(analCofing['loadRoi'])
        if analCofing['studyExtension']:
            self.studyExtension()
        if analCofing['doSed']:
            self.calcSed(analCofing['loadRoi'], analCofing['loadSed'], analCofing['mergeBins'])
        if analCofing['doLC']:
            self.calcLC(analCofing['loadRoi'], analCofing['lcBins'],
                        self.gta._config['selection']['tmin'],
                        self.gta._config['selection']['tmax'],
                        analCofing['threads'])

        return

    def performFit(self, redoSetup):

        # This is here only because at the moment FermiPy fails if ccube.fits exists
        # It is fixed in the latest development version of FermiPy, but in the meantime
        # need to do this
        if os.path.exists(self.gta.outdir + '/ccube.fits'):
            os.remove(self.gta.outdir + '/ccube.fits')

        # If you change the config file, you need to overwrite the setup (see below)
        self.gta.setup(overwrite=redoSetup)

        opt1 = self.gta.optimize()

        deleted_sourcesTS = self.gta.delete_sources(minmax_ts=[-1, 1])
        deleted_sourcesNpred = self.gta.delete_sources(minmax_npred=[0, 2])

        self.gta.print_roi()

        # Free Normalization of all Sources within 5 deg of ROI center
        self.gta.free_sources(distance=5.0, pars='norm')

        # Free all parameters of the source in question
        self.gta.free_source(self.gta.roi.sources[0].name)

        # Free sources with TS > 10
        self.gta.free_sources(minmax_ts=[10, None], pars='norm')

        # Free all parameters of isotropic and galactic diffuse components
        self.gta.free_source('galdiff')
        self.gta.free_source('isodiff')

        fit1 = self.gta.fit()

        self.gta.print_roi()
        print(self.gta.roi[self.gta.roi.sources[0].name])
        self.gta.write_roi('fit_%s_%s' % (self.targetShort, self.titleSuffix), make_plots=True)

        fixed_sources = self.gta.free_sources(free=False)

    def plotMaps(self, loadRoi):

        # This is here only because at the moment FermiPy fails if ccube.fits exists
        # It is fixed in the latest development version of FermiPy, but in the meantime
        # need to do this
        if os.path.exists(self.gta.outdir + '/ccube.fits'):
            os.remove(self.gta.outdir + '/ccube.fits')

        if loadRoi:
            self.gta.load_roi('fit_%s_%s' % (self.targetShort, self.titleSuffix))

        resid = self.gta.residmap(self.target, model={'SpatialModel': 'PointSource', 'Index': 2.0})
        fig = plt.figure(figsize=(14, 6))
        ROIPlotter(resid['sigma'], roi=self.gta.roi).plot(vmin=-5, vmax=5,
                                                          levels=[-5, -3, 3, 5, 7, 9],
                                                          subplot=121, cmap='RdBu_r')
        plt.gca().set_title('Significance')
        ROIPlotter(resid['excess'], roi=self.gta.roi).plot(vmin=-200, vmax=200,
                                                           subplot=122, cmap='RdBu_r')
        plt.gca().set_title('Excess Counts')
        plt.savefig(self.gta.outdir + '/resid.pdf')

        plt.clf()

        resid_noTarget = self.gta.residmap('without' + self.targetShort,
                                           model={'SpatialModel': 'PointSource', 'Index': 2.0},
                                           exclude=[self.gta.roi.sources[0].name])

        fig = plt.figure(figsize=(14, 6))
        ROIPlotter(resid_noTarget['sigma'], roi=self.gta.roi).plot(vmin=-5, vmax=5,
                                                                   levels=[-5, -3, 3, 5, 7, 9],
                                                                   subplot=121, cmap='RdBu_r')
        plt.gca().set_title('Significance')
        ROIPlotter(resid_noTarget['excess'], roi=self.gta.roi).plot(vmin=-200, vmax=200,
                                                                    subplot=122, cmap='RdBu_r')
        plt.gca().set_title('Excess Counts')
        plt.savefig(self.gta.outdir + '/residNo%s.pdf' % self.targetShort)

        plt.clf()

        tsmapIndex2 = self.gta.tsmap(self.gta.roi.sources[0].name,
                                     model={'SpatialModel': 'PointSource', 'Index': 2.0})
        tsmapIndexFit = self.gta.tsmap(self.gta.roi.sources[0].name,
                                       model={'SpatialModel': 'PointSource', 'Index': 1.66})

        o2 = tsmapIndex2
        oFit = tsmapIndexFit

        fig = plt.figure(figsize=(14, 6))
        ROIPlotter(oFit['sqrt_ts'], roi=self.gta.roi).plot(vmin=0, vmax=5,
                                                           levels=[3, 5, 7, 9],
                                                           subplot=121, cmap='magma')
        plt.gca().set_title('sqrt(TS) Index from fit')
        ROIPlotter(o2['sqrt_ts'], roi=self.gta.roi).plot(vmin=0, vmax=5,
                                                         levels=[3, 5, 7, 9],
                                                         subplot=122, cmap='magma')
        plt.gca().set_title('sqrt(TS) Index=2')
        plt.savefig(self.gta.outdir + '/tsMap.pdf')

        plt.clf()

        tsmap_noTarget = self.gta.tsmap('without' + self.targetShort,
                                        model={'SpatialModel': 'PointSource', 'Index': 2.0},
                                        exclude=[self.gta.roi.sources[0].name])

        fig = plt.figure(figsize=(6, 6))
        ROIPlotter(tsmap_noTarget['sqrt_ts'], roi=self.gta.roi).plot(vmin=0, vmax=5,
                                                                     levels=[3, 5, 7, 9],
                                                                     subplot=111, cmap='magma')
        plt.gca().set_title('sqrt(TS)')
        plt.savefig(self.gta.outdir + '/tsMapNo%s.pdf' % self.targetShort)

        plt.clf()

    def studyExtension(self):

        ext_gauss = self.gta.extension(self.gta.roi.sources[0].name,
                                       free_background=True, free_radius=2.0,
                                       make_plots=True)

        gta.write_roi('ext_gauss_fit')

        plt.figure(figsize=(8, 6))
        plt.plot(ext_gauss['width'], ext_gauss['dloglike'], marker='o')
        plt.gca().set_xlabel('Width [deg]', fontsize=18)
        plt.gca().set_ylabel('Delta Log-Likelihood', fontsize=18)
        plt.gca().axvline(ext_gauss['ext'])
        plt.gca().axvspan(ext_gauss['ext'] - ext_gauss['ext_err_lo'],
                          ext_gauss['ext'] + ext_gauss['ext_err_hi'],
                          alpha=0.2, label='1ES0033', color='b')

        plt.annotate(r'TS$_{\mathrm{ext}}$ = %.2f\nR$_{68}$ = %.3f $\pm$ %.3f' %
                     (ext_gauss['ts_ext'], ext_gauss['ext'], ext_gauss['ext_err']),
                     xy=(0.05, 0.05), xycoords='axes fraction', fontsize=15)

        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.grid(True)
        plt.gca().set_axisbelow(True)
        plt.tight_layout()
        plt.savefig('studyExtension.pdf')

    def calcSed(self, loadRoi, loadSed, mergeBins):

        # This is here only because at the moment FermiPy fails if ccube.fits exists
        # It is fixed in the latest development version of FermiPy, but in the meantime
        # need to do this
        if os.path.exists(self.gta.outdir + '/ccube.fits'):
            os.remove(self.gta.outdir + '/ccube.fits')

        if loadRoi:
            self.gta.load_roi('fit_%s_%s' % (self.targetShort, self.titleSuffix))

        if mergeBins:
            emin = self.gta.config['selection']['emin']
            emax = self.gta.config['selection']['emax']
            logemin = np.log10(emin)
            logemax = np.log10(emax)
            nBins = np.round(self.gta.config['binning']['binsperdec']*np.log10(emax / emin))
            nBins = int(nBins)

            logEnergies = np.linspace(logemin, logemax, nBins + 1)
            newLogEnergies = logEnergies[[0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 22, 25]]

            sed = self.gta.sed(self.target, loge_bins=newLogEnergies,
                               make_plots=True, use_local_index=True)

        else:
            sed = self.gta.sed(self.target)

        E = np.array(sed['model_flux']['energies'])
        dnde = np.array(sed['model_flux']['dnde'])
        dnde_hi = np.array(sed['model_flux']['dnde_hi'])
        dnde_lo = np.array(sed['model_flux']['dnde_lo'])

        # Should change the plotting below based on the spectrum type.
        # At the moment the code below supports only a simple power law
        sourceType = self.gta.roi[self.gta.roi.sources[0].name]['SpectrumType']

        # i_norm = np.where(self.gta.roi[self.gta.roi.sources[0].name]['param_names'] == 'Prefactor')
        # print(i_norm, self.gta.roi[self.gta.roi.sources[0].name]['param_values'][i_norm])
        # i_norm = np.where(self.gta.roi[self.gta.roi.sources[0].name]['param_names'] == 'norm')
        # print(i_norm, self.gta.roi[self.gta.roi.sources[0].name]['param_values'][i_norm])
        # norm = float(self.gta.roi[self.gta.roi.sources[0].name]['param_values'][i_norm])
        # normErr = float(self.gta.roi[self.gta.roi.sources[0].name]['param_errors'][i_norm])

        # FIXME - still not working!!
        # i_idx = np.where(self.gta.roi[self.gta.roi.sources[0].name]['param_names'] == 'Index')
        # print(i_idx)
        # print(type(i_idx))
        # print(len(i_idx))
        # if i_idx:
        #     idx = float(self.gta.roi[self.gta.roi.sources[0].name]['param_values'][i_idx])
        #     idxErr = float(self.gta.roi[self.gta.roi.sources[0].name]['param_errors'][i_idx])

        plt.figure(figsize=(15, 6))
        plt.loglog(E, dnde, 'k--')
        plt.loglog(E, dnde_hi, 'k')
        plt.loglog(E, dnde_lo, 'k')
        plt.errorbar(np.array(sed['e_ctr']),
                     sed['dnde'],
                     yerr=sed['dnde_err'], fmt='o')
        plt.xlabel('E [MeV]', fontsize=18)
        plt.ylabel(r'dN/dE [MeV$^{-1}$ cm$^{-2}$ s$^{-1}$]', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title('Fermi spectrum %s (%s)' % (self.titleSuffix, sourceType), fontsize=15, y=1.02)
        # if idx:
        #     plt.text(0.95, 0.95, r'$\Gamma$ = %1.3f $\pm$ %1.3f' % (idx, abs(idxErr)),
        #              verticalalignment='top', horizontalalignment='right',
        #              transform=plt.gca().transAxes,
        #              color='black', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.gta.outdir + '/sed.pdf')

        plt.clf()

        self.gta.write_roi('sed_%s_%s' % (self.targetShort, self.titleSuffix), make_plots=True)

    def calcLC(self, loadRoi, lcBins, tmin, tmax, nthread):

        # This is here only because at the moment FermiPy fails if ccube.fits exists
        # It is fixed in the latest development version of FermiPy, but in the meantime
        # need to do this
        if os.path.exists(self.gta.outdir + '/ccube.fits'):
            os.remove(self.gta.outdir + '/ccube.fits')

        if loadRoi:
            self.gta.load_roi('fit_%s_%s' % (self.targetShort, self.titleSuffix))

        nWeekBins = int(lcBins/7)
        timeBins = list(np.arange(tmin, tmax, 86400.*lcBins))

        lc = self.gta.lightcurve(self.gta.roi.sources[0].name,
                                 binsz=86400.*lcBins,
                                 multithread=True, nthread=nthread,
                                 use_scaled_srcmap=True)
        # To use my time binning (which fails sometimes...
        # lc = self.gta.lightcurve(self.gta.roi.sources[0].name,
        #                          time_bins=timeBins,
        #                          multithread=True, nthread=nthread,
        #                          use_scaled_srcmap=True)

        plt.figure(figsize=(15, 6))
        plt.plot(np.sqrt(lc['npred'])/lc['npred'], lc['flux_err']/lc['flux'], 'ko')
        plt.xlabel(r'$\sqrt{\mathrm{N}_\mathrm{pred}}$/$\mathrm{N}_\mathrm{pred}$', fontsize=18)
        plt.ylabel(r'$\Delta$F/F', fontsize=18)
        plt.savefig(self.gta.outdir + '/fluxVsNpred.pdf')

        plt.clf()

        plt.figure(figsize=(15, 6))
        plt.errorbar(lc['tmax_mjd'], lc['flux'], yerr=lc['flux_err'], fmt='o')
        plt.xlabel('%d-week bins [MJD]' % nWeekBins, fontsize=18)
        plt.ylabel(r'Flux [cm$^{-2}$ s$^{-1}$]', fontsize=18)

        conF, conFcov = curve_fit(self.conFunc, lc['tmax_mjd'],
                                  lc['flux'], [lc['flux'][0]], lc['flux_err'])
        conChi2 = np.sum((((lc['flux'] - self.conFunc(lc['tmax_mjd'], *conF)) ** 2) /
                          (lc['flux_err']**2)))
        plt.plot(lc['tmax_mjd'], len(lc['tmax_mjd'])*list(conF), 'b--',
                 label=(r'Constant Fit - '
                        r'$\chi^2$/ndf = {goodFit:f}').format(goodFit=(conChi2 /
                                                                       (len(lc['tmax_mjd'])) - 1)))

        plt.legend(loc=1, fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title('Fermi light curve - flux - %s' % self.titleSuffix, fontsize=15, y=1.02)
        plt.tight_layout()

        plt.savefig(self.gta.outdir + '/lightcurve_%s_flux.pdf' % self.targetShort)
        plt.clf()

        plt.figure(figsize=(15, 6))
        plt.errorbar(lc['tmax_mjd'], abs(lc['param_values'][:, 1]),
                     yerr=abs(lc['param_errors'][:, 1]), fmt='o', color='orange')
        plt.xlabel('%d-week bins [MJD]' % nWeekBins, fontsize=18)
        plt.ylabel(r'Index', fontsize=18)

        conF, conFcov = curve_fit(self.conFunc, lc['tmax_mjd'],
                                  abs(lc['param_values'][:, 1]),
                                  [lc['param_values'][:, 1][0]],
                                  abs(lc['param_errors'][:, 1]))
        conChi2 = np.sum(((abs(lc['param_values'][:, 1]) -
                         self.conFunc(lc['tmax_mjd'], *conF))**2) /
                         (abs(lc['param_errors'][:, 1])**2))
        plt.plot(lc['tmax_mjd'], len(lc['tmax_mjd'])*list(conF), 'b--',
                 label=(r'Constant Fit - '
                        r'$\chi^2$/ndf = {goodFit:f}').format(goodFit=(conChi2 /
                                                                       (len(lc['tmax_mjd'])) - 1)))

        plt.legend(loc=1, fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title('Fermi light curve - index - %s' % self.titleSuffix, fontsize=15, y=1.02)
        plt.tight_layout()

        plt.savefig(self.gta.outdir + '/lightcurve_%s_index.pdf' % self.targetShort)
        plt.clf()

        plt.figure(figsize=(15, 6))
        plt.errorbar(lc['tmax_mjd'], lc['ts'], fmt='o', color='forestgreen')
        plt.xlabel('%d-week bins [MJD]' % nWeekBins, fontsize=18)
        plt.ylabel(r'TS', fontsize=18)

        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.title('Fermi light curve - test statisitcs - %s' % self.titleSuffix,
                  fontsize=15, y=1.02)
        plt.tight_layout()

        plt.savefig(self.gta.outdir + '/lightcurve_%s_ts.pdf' % self.targetShort)
        plt.clf()

    def conFunc(self, x, a):
        return a

    def prepareSource(self, configFermiPy, configAnal,
                      configSources, dirNow, sourceName, sourceTitle, overwrite=False):
        """
        Prepare directory and configuration file for a source
        given in sourceName. The directory name is given in dirNow.
        The template configuration files are used as a basis for the
        configuration for this specific source.
        """

        if overwrite:
            shutil.rmtree(os.path.join(os.getcwd(), dirNow), ignore_errors=True)

        with open(configFermiPy, 'r') as fConfig:
            fermiPyCofing = yaml.safe_load(fConfig)
        with open(configAnal, 'r') as fAnal:
            analCofing = yaml.safe_load(fAnal)

        fermiPyCofing['selection']['target'] = sourceTitle
        fermiPyCofing['fileio']['outdir'] = sourceName
        analCofing['title'] = sourceTitle

        if sourceName in configSources:
            if 'lcBins' in configSources[sourceName]:
                if isinstance(analCofing['lcBins'], list):
                    analCofing['lcBins'] = configSources[sourceName]['lcBins']
                elif analCofing['lcBins'] < configSources[sourceName]['lcBins']:
                    analCofing['lcBins'] = configSources[sourceName]['lcBins']

        if not os.path.exists(os.path.join(os.getcwd(), dirNow)):
            os.mkdir(os.path.join(os.getcwd(), dirNow))
        if not os.path.exists(os.path.join(os.getcwd(), dirNow, sourceName)):
            os.mkdir(os.path.join(os.getcwd(), dirNow, sourceName))

        newFermiPyFile = os.path.join(os.getcwd(), dirNow, 'configFermiPy.yaml')
        newAnalFile = os.path.join(os.getcwd(), dirNow, 'configAnal.yaml')

        with open(newFermiPyFile, 'w') as fOutYml:
            yaml.dump(fermiPyCofing, fOutYml, allow_unicode=True, default_flow_style=False)
        with open(newAnalFile, 'w') as fOutYml:
            yaml.dump(analCofing, fOutYml, allow_unicode=True, default_flow_style=False)

        os.symlink(os.path.join(os.getcwd(), 'anal.py'),
                   os.path.join(os.getcwd(), dirNow, 'anal.py'))
        os.symlink(os.path.join(os.getcwd(), 'runAnal.sh'),
                   os.path.join(os.getcwd(), dirNow, 'runAnal.sh'))

        return

    def submitSource(self, configFermiPy, configAnal, dirNow, sourceName):
        """
        Submit analysis for source in dirNow
        """

        origWdir = os.getcwd()
        os.chdir(os.path.join(origWdir, dirNow))
        wdir = os.getcwd()
        logFile = os.path.join(wdir, sourceName, '{}.log'.format(sourceName))

        with open(configAnal, 'r') as fAnal:
            analCofing = yaml.safe_load(fAnal)
        thread = ''
        if analCofing['doLC']:
            thread = ' -R y -pe multicore {}'.format(analCofing['threads'])

        # high priority: -P cta_high
        cmd = ('qsub -js 9 -N {jobName:s} -l h_cpu={cpuTime:s} {thread:s} '
               '-l tmpdir_size=15G -l h_rss=4G -V -o {log:s} -e {err:s} '
               '"runAnal.sh" {wdir:s} {configFermiPy:s} {configAnal:s}'
               '').format(jobName='j_' + dirNow,
                          cpuTime=analCofing['cpuTime'],
                          thread=thread,
                          log=logFile,
                          err=logFile,
                          wdir=wdir,
                          configFermiPy=configFermiPy,
                          configAnal=configAnal)

        subprocess.call(cmd, shell=True)

        os.chdir(origWdir)

        return

    def argparsing(self):

        parser = argparse.ArgumentParser(description=('Run Fermi analysis following '
                                                      'the settings in the config file.'))
        parser.add_argument('mode', choices=['analyze', 'lightcurve',
                                             'prepareSources', 'submitSources'],
                            help='Run the normal analysis or produce light-curves for each year')
        parser.add_argument('configFile', action='store',
                            help='The config file to use')
        parser.add_argument('analCofing', action='store',
                            help='The analysis config file to use')

        # parser.set_defaults(mode='analyze')

        args = parser.parse_args()
        return args


if __name__ == '__main__':

    fermiAnal = fermiAnal()
    # Parse the command line
    args = fermiAnal.argparsing()

    # fermiAnal.analyze(args.configFile, args.analCofing)

    if args.mode == 'analyze':
        fermiAnal.analyze(args.configFile, args.analCofing)
    elif args.mode == 'prepareSources':
        sources = {
                   # '1ES0033': '1ES 0033+595',
                   # '1ES0502': '1ES 0502+675',
                   # '1ES1011': '1ES 1011+496',
                   # '1ES1218': '1ES 1218+304',
                   # '1ES0229': '1ES 0229+200', # Set binsperdec : 4 for this source!!
                   # 'RGBJ0710': 'RGB J0710+591', # Set binsperdec : 5 for this source!!
                   # 'PG1553':  'PG 1553+113',
                   # 'PKS1424': 'PKS 1424+240'
                   'TON599': 'TON 0599'
                   }
        configSources = {
                         '1ES0033': {'lcBins': 56},
                         '1ES0502': {'lcBins': 56},
                         # '1ES1011': {'lcBins': 7},
                         '1ES1218': {'lcBins': 56},
                         '1ES0229': {'lcBins': 140},  # used to be 56
                         'RGBJ0710': {'lcBins': 140}  # used to be 56
                         }
        with open(args.analCofing, 'r') as fAnal:
            analCofing = yaml.safe_load(fAnal)

        for sourceNowShort, sourceNowLong in sources.items():
            if isinstance(analCofing['lcBins'], list):
                for lcBinNow in analCofing['lcBins']:
                    lcDirNow = '{}dayBins_{}'.format(lcBinNow, sourceNowShort)
                    configSources[sourceNowShort] = {'lcBins': lcBinNow}
                    fermiAnal.prepareSource(args.configFile, args.analCofing,
                                            configSources, lcDirNow, sourceNowShort,
                                            sourceNowLong, True)
            else:
                fermiAnal.prepareSource(args.configFile, args.analCofing,
                                        configSources, sourceNowShort,
                                        sourceNowShort, sourceNowLong, True)
    elif args.mode == 'submitSources':
        sources = {
                   # '1ES0033': '1ES 0033+595',
                   # '1ES0502': '1ES 0502+675',
                   # '1ES1011': '1ES 1011+496',
                   # '1ES1218': '1ES 1218+304',
                   # '1ES0229': '1ES 0229+200',
                   # 'RGBJ0710': 'RGB J0710+591',
                   # 'PG1553':  'PG 1553+113',
                   # 'PKS1424': 'PKS 1424+240'
                   'TON599': 'TON 0599'
                   }
        with open(args.analCofing, 'r') as fAnal:
            analCofing = yaml.safe_load(fAnal)

        for sourceNowShort in sources.keys():
            if isinstance(analCofing['lcBins'], list):
                for lcBinNow in analCofing['lcBins']:
                    lcDirNow = '{}dayBins_{}'.format(lcBinNow, sourceNowShort)
                    configFermiPy = os.path.join(os.getcwd(), lcDirNow, 'configFermiPy.yaml')
                    configAnal = os.path.join(os.getcwd(), lcDirNow, 'configAnal.yaml')
                    fermiAnal.submitSource(configFermiPy, configAnal, lcDirNow, sourceNowShort)
            else:
                configFermiPy = os.path.join(os.getcwd(), sourceNowShort, 'configFermiPy.yaml')
                configAnal = os.path.join(os.getcwd(), sourceNowShort, 'configAnal.yaml')
                fermiAnal.submitSource(configFermiPy, configAnal, sourceNowShort, sourceNowShort)
